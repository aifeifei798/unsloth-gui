"""数据处理：把任意来源的数据集制成统一训练数据.

统一 schema（训练只认这个）：
    instruction: str  # 输入/指令（必填映射）
    think: str        # 思维链（可选映射，没有就空字符串）
    output: str       # 回复（必填映射，空回复的行会被丢弃并计数）

产物（local_data/processed/<name>/）：
    data.jsonl        # 人可读的统一数据
    hf_dataset/       # arrow 落盘，训练时 load_from_disk 秒载
    manifest.json     # 来源、映射、行数、时间
外加 datasets_config/<name>.json（processed=true），训练 Tab 自动出现。
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT, DatasetConfig

PROCESSED_ROOT = PROJECT_ROOT / "local_data" / "processed"
UPLOAD_ROOT = PROJECT_ROOT / "local_data" / "uploads"
DATASETS_CONFIG_DIR = PROJECT_ROOT / "datasets_config"

NO_THINK = "(无 / 不映射)"

UNIFIED_WITH_THINK = (
    "### Instruction:\n{instruction}\n\n"
    "### Thinking:\n{think}\n\n"
    "### Response:\n{output}"
)
UNIFIED_NO_THINK = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

SCHEMA_VERSION = 1


def sanitize_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", name).strip("_")
    if not name:
        name = "processed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return name[:80]


def load_raw_dataset(kind: str, hf_id: str = "", split: str = "train",
                     local_path: str = "", existing: Optional[DatasetConfig] = None):
    """按来源加载原始数据集（未格式化）. kind: upload | hf | existing."""
    from datasets import load_dataset, load_from_disk

    if kind == "existing":
        if existing is None:
            raise ValueError("请选择一个已有数据集配置。")
        p = Path(existing.resolved_dataset_id())
        if p.exists():
            if p.is_dir():
                ds = load_from_disk(str(p))
            elif p.suffix in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=str(p), split="train")
            elif p.suffix == ".csv":
                ds = load_dataset("csv", data_files=str(p), split="train")
            elif p.suffix in (".parquet",):
                ds = load_dataset("parquet", data_files=str(p), split="train")
            elif p.suffix == ".txt":
                ds = load_dataset("text", data_files=str(p), split="train")
            else:
                raise ValueError(f"不支持的本地格式: {p}（支持目录/jsonl/json/csv/parquet/txt）")
        else:
            if existing.is_local:
                raise FileNotFoundError(f"本地路径未找到: {p}")
            ds = load_dataset(existing.dataset_id, split=existing.split)
    elif kind == "hf":
        hf_id = (hf_id or "").strip()
        if not hf_id:
            raise ValueError("请填写 HuggingFace 数据集 ID（如 yahma/alpaca-cleaned）。")
        ds = load_dataset(hf_id, split=(split or "train").strip() or "train")
    elif kind == "upload":
        if not local_path or not Path(local_path).is_file():
            raise ValueError("请先上传文件（支持 .jsonl / .json / .csv / .parquet / .txt）。")
        p = Path(local_path)
        if p.suffix in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=str(p), split="train")
        elif p.suffix == ".csv":
            ds = load_dataset("csv", data_files=str(p), split="train")
        elif p.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(p), split="train")
        elif p.suffix == ".txt":
            ds = load_dataset("text", data_files=str(p), split="train")
        else:
            raise ValueError(f"不支持的上传格式: {p.suffix}")
    else:
        raise ValueError(f"未知来源: {kind}")

    if hasattr(ds, "keys") and not hasattr(ds, "map"):  # DatasetDict
        want = (split or "").strip()
        ds = ds[want] if want in ds else ds[list(ds.keys())[0]]
    return ds


def persist_upload(src_path: str) -> str:
    """把 Gradio 临时上传文件拷到 local_data/uploads/ 持久化."""
    src = Path(src_path)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_ROOT / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def inspect_text(kind: str, hf_id: str = "", split: str = "train",
                 local_path: str = "", existing: Optional[DatasetConfig] = None,
                 n_sample: int = 2) -> tuple[str, dict]:
    """返回（展示文本， 状态dict供生成步骤复用）."""
    ds = load_raw_dataset(kind, hf_id, split, local_path, existing)
    cols = list(ds.column_names)
    lines = [
        f"总行数: {len(ds)}",
        f"列名 ({len(cols)}): {', '.join(cols)}",
        "-" * 60,
    ]
    for i in range(min(n_sample, len(ds))):
        row = ds[i]
        preview = {k: (str(v)[:300] + "…" if len(str(v)) > 300 else v) for k, v in row.items()}
        lines.append(f"[第 {i+1} 行]\n" + json.dumps(preview, ensure_ascii=False, indent=1))
        lines.append("-" * 60)
    state = {"kind": kind, "hf_id": hf_id, "split": split,
             "local_path": local_path,
             "existing_name": existing.display_name if existing else "",
             "columns": cols, "n_rows": len(ds)}
    return "\n".join(lines), state


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def generate_unified(state: dict, instruction_col: str, think_col: str,
                     output_col: str, output_name: str,
                     progress=None) -> tuple[str, str, str]:
    """生成统一数据。返回 (状态文本, 预览文本, 新数据集display_name)."""
    if not state or not state.get("columns"):
        raise ValueError("请先点「1. 读取列信息」。")
    cols = state["columns"]
    if not instruction_col or instruction_col not in cols:
        raise ValueError("请选择 instruction 输入列。")
    if not output_col or output_col not in cols:
        raise ValueError("请选择 output 回复列。")
    if instruction_col == output_col:
        raise ValueError("instruction 列和 output 列不能是同一列。")
    use_think = bool(think_col and think_col != NO_THINK)
    if use_think and think_col not in cols:
        raise ValueError(f"think 列 '{think_col}' 不存在。")
    if use_think and think_col in (instruction_col, output_col):
        raise ValueError("think 列不能和 instruction/output 列相同。")

    name = sanitize_name(output_name)
    cfg_path = DATASETS_CONFIG_DIR / f"{name}.json"
    if cfg_path.exists():
        raise ValueError(f"数据集名 '{name}' 已存在（{cfg_path.name}），请换个名字。")

    from .config import load_datasets_config, find_by_name
    try:
        if find_by_name(load_datasets_config(), name) is not None:
            raise ValueError(f"展示名 '{name}' 已被占用，请换个名字。")
    except FileNotFoundError:
        pass  # 配置目录还没建，一会儿一起建

    if progress is not None:
        try:
            progress(0.1, desc="加载原始数据...")
        except Exception:
            pass
    from .config import find_by_name as _find
    existing = None
    if state.get("kind") == "existing" and state.get("existing_name"):
        try:
            existing = _find(load_datasets_config(), state["existing_name"])
        except Exception:
            existing = None
    ds = load_raw_dataset(state["kind"], state.get("hf_id", ""), state.get("split", "train"),
                          state.get("local_path", ""), existing)

    if progress is not None:
        try:
            progress(0.4, desc="映射列并清洗...")
        except Exception:
            pass
    rows: list[dict] = []
    dropped_empty = 0
    for row in ds:
        out = _cell(row.get(output_col))
        if not out:
            dropped_empty += 1
            continue
        rows.append({
            "instruction": _cell(row.get(instruction_col)),
            "think": _cell(row.get(think_col)) if use_think else "",
            "output": out,
        })
    if not rows:
        raise ValueError("有效行数为 0（所有行的 output 列都是空的），请检查列映射。")

    if progress is not None:
        try:
            progress(0.7, desc="落盘统一数据...")
        except Exception:
            pass
    out_dir = PROCESSED_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from datasets import Dataset
    hf_ds = Dataset.from_list(rows)
    arrow_dir = out_dir / "hf_dataset"
    if arrow_dir.exists():
        shutil.rmtree(arrow_dir)
    hf_ds.save_to_disk(str(arrow_dir))

    manifest = {
        "name": name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {k: state.get(k) for k in ("kind", "hf_id", "split", "existing_name")},
        "mapping": {"instruction": instruction_col,
                    "think": think_col if use_think else None,
                    "output": output_col},
        "rows_total": len(ds),
        "rows_kept": len(rows),
        "rows_dropped_empty_output": dropped_empty,
        "schema_version": SCHEMA_VERSION,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    template = UNIFIED_WITH_THINK if use_think else UNIFIED_NO_THINK
    input_columns = {"instruction": "instruction", "output": "output"}
    if use_think:
        input_columns = {"instruction": "instruction", "think": "think", "output": "output"}
    cfg = {
        "display_name": name,
        "dataset_id": f"./local_data/processed/{name}/hf_dataset",
        "is_local": True,
        "processed": True,
        "schema_version": SCHEMA_VERSION,
        "prompt_template": template,
        "input_columns": input_columns,
    }
    DATASETS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if progress is not None:
        try:
            progress(1.0, desc="完成")
        except Exception:
            pass
    status = (f"✅ 已生成统一训练数据 '{name}'：原始 {len(ds)} 行 → 保留 {len(rows)} 行"
              + (f"（丢弃空回复 {dropped_empty} 行）" if dropped_empty else ""))
    prev_lines = [status, f"模板: {'含 Thinking' if use_think else '无 Thinking'}", "-" * 60]
    for i, r in enumerate(rows[:2]):
        filled = template.format(**r)
        if len(filled) > 1000:
            filled = filled[:1000] + "\n…(截断)"
        prev_lines.append(f"[统一后样本 {i+1}]\n{filled}\n" + "-" * 60)
    prev_lines.append("现在可以去「训练」Tab 选中它开始训练（点刷新列表）。")
    return status, "\n".join(prev_lines), name
