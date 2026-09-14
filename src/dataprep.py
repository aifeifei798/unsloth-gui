"""数据处理：把任意来源的数据集制成统一训练数据.

统一 schema（训练只认这个，四个角色都可多选，多列按顺序换行拼接）：
    instruction: str  # 指令/问题（至少映射 1 列）
    input: str        # 补充输入/上下文（可选）
    think: str        # 思维链（可选）
    output: str       # 回复（至少映射 1 列，空回复的行会被丢弃并计数）

模板按实际映射动态组装（没映射 input/think 就没有对应段落，不浪费 token）。
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

SCHEMA_VERSION = 2

ROLES = ("instruction", "input", "think", "output")
ROLE_LABEL = {"instruction": "instruction 输入", "input": "input 上下文",
              "think": "think 思维链", "output": "output 回复"}


def build_template(use_input: bool, use_think: bool) -> tuple[str, dict]:
    """按实际映射组装模板与 input_columns."""
    parts = ["### Instruction:\n{instruction}"]
    columns = {"instruction": "instruction"}
    if use_input:
        parts.append("### Input:\n{input}")
        columns["input"] = "input"
    if use_think:
        parts.append("### Thinking:\n{think}")
        columns["think"] = "think"
    parts.append("### Response:\n{output}")
    columns["output"] = "output"
    return "\n\n".join(parts), columns


def sanitize_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", name).strip("_")
    if not name:
        name = "processed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return name[:80]


def load_raw_dataset(kind: str, hf_id: str = "", split: str = "train",
                     local_path: str = "", existing: Optional[DatasetConfig] = None,
                     streaming: bool = False):
    """按来源加载原始数据集（未格式化）. kind: upload | hf | existing.
    streaming=True 时远端走流式（只用于预览取行，不可 len()/索引）."""
    from datasets import load_dataset, load_from_disk

    if kind == "existing":
        if existing is None:
            raise ValueError("请选择一个已有数据集配置。")
        p = Path(existing.resolved_dataset_id())
        if p.exists():
            if p.is_dir():
                ds = load_from_disk(str(p))  # 本地落盘本来就快，不走流式
            elif p.suffix in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=str(p), split="train", streaming=streaming)
            elif p.suffix == ".csv":
                ds = load_dataset("csv", data_files=str(p), split="train", streaming=streaming)
            elif p.suffix in (".parquet",):
                ds = load_dataset("parquet", data_files=str(p), split="train", streaming=streaming)
            elif p.suffix == ".txt":
                ds = load_dataset("text", data_files=str(p), split="train", streaming=streaming)
            else:
                raise ValueError(f"不支持的本地格式: {p}（支持目录/jsonl/json/csv/parquet/txt）")
        else:
            if existing.is_local:
                raise FileNotFoundError(f"本地路径未找到: {p}")
            ds = load_dataset(existing.dataset_id, split=existing.split, streaming=streaming)
    elif kind == "hf":
        hf_id = (hf_id or "").strip()
        if not hf_id:
            raise ValueError("请填写 HuggingFace 数据集 ID（如 yahma/alpaca-cleaned）。")
        ds = load_dataset(hf_id, split=(split or "train").strip() or "train", streaming=streaming)
    elif kind == "upload":
        if not local_path or not Path(local_path).is_file():
            raise ValueError("请先上传文件（支持 .jsonl / .json / .csv / .parquet / .txt）。")
        p = Path(local_path)
        if p.suffix in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=str(p), split="train", streaming=streaming)
        elif p.suffix == ".csv":
            ds = load_dataset("csv", data_files=str(p), split="train", streaming=streaming)
        elif p.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(p), split="train", streaming=streaming)
        elif p.suffix == ".txt":
            ds = load_dataset("text", data_files=str(p), split="train", streaming=streaming)
        else:
            raise ValueError(f"不支持的上传格式: {p.suffix}")
    else:
        raise ValueError(f"未知来源: {kind}")

    if hasattr(ds, "keys") and not hasattr(ds, "map"):  # DatasetDict
        want = (split or "").strip()
        ds = ds[want] if want in ds else ds[list(ds.keys())[0]]
    return ds


def peek_source(kind: str, hf_id: str = "", split: str = "train",
                local_path: str = "", existing: Optional[DatasetConfig] = None,
                ) -> tuple[list, Optional[dict], Optional[int], bool]:
    """只取 1 行做映射预览：远端走 streaming 秒开（不下载全量），本地直接读。
    返回 (列名, 首行|None, 总行数|None, 是否流式)。全量数据等点生成时再拉取。"""
    split = (split or "train").strip() or "train"
    remote = kind == "hf" or (
        kind == "existing" and existing is not None
        and not Path(existing.resolved_dataset_id()).exists()
    )
    if remote:
        from datasets import load_dataset
        ds_id = hf_id.strip() if kind == "hf" else existing.dataset_id
        if not ds_id:
            raise ValueError("请填写 HuggingFace 数据集 ID。")
        ds = load_dataset(ds_id, split=split, streaming=True)
        cols = list(ds.column_names or [])
        try:
            row = next(iter(ds))
        except StopIteration:
            row = None
        if not cols and row:  # 流式下 column_names 可能为空，用首行 keys 兜底
            cols = list(row.keys())
        return cols, row, None, True
    ds = load_raw_dataset(kind, hf_id, split, local_path, existing)
    cols = list(ds.column_names)
    row = ds[0] if len(ds) > 0 else None
    return cols, row, len(ds), False


def persist_upload(src_path: str) -> str:
    """把 Gradio 临时上传文件拷到 local_data/uploads/ 持久化."""
    src = Path(src_path)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_ROOT / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def inspect_text(kind: str, hf_id: str = "", split: str = "train",
                 local_path: str = "", existing: Optional[DatasetConfig] = None) -> tuple[str, dict]:
    """只取 1 行预览列信息（远端流式秒开）；全量数据等生成时再拉取。"""
    cols, row, n_rows, streamed = peek_source(kind, hf_id, split, local_path, existing)
    if not cols:
        raise ValueError("未能读到任何列，请检查来源与切分名。")
    count_line = (f"总行数: 未知（流式预览，点生成时全量拉取）" if streamed
                  else f"总行数: {n_rows}")
    lines = [
        count_line,
        f"列名 ({len(cols)}): {', '.join(cols)}",
        "-" * 60,
    ]
    if row is None:
        lines.append("(空数据集，没有可预览的行)")
    else:
        preview = {k: (str(v)[:300] + "…" if len(str(v)) > 300 else v) for k, v in row.items()}
        lines.append("[预览第 1 行]\n" + json.dumps(preview, ensure_ascii=False, indent=1))
        lines.append("-" * 60)
    state = {"kind": kind, "hf_id": hf_id, "split": split,
             "local_path": local_path,
             "existing_name": existing.display_name if existing else "",
             "columns": cols, "n_rows": n_rows, "streamed": streamed}
    return "\n".join(lines), state


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    return [c for c in v if c]


def _merge(row: dict, cols: list) -> str:
    """多列按顺序换行拼接，跳过空值."""
    return "\n".join(_cell(row.get(c)) for c in cols if _cell(row.get(c)))


def generate_unified(state: dict, instruction_cols, input_cols,
                     think_cols, output_cols, output_name: str,
                     fixed_instruction: str = "", progress=None) -> tuple[str, str, str]:
    """生成统一数据。四个角色都可多选；instruction 还支持手写固定文本
   （数据里没有对应列时用它，所有行共用；同时映射了列则做前缀拼在前面）.
    返回 (状态文本, 预览文本, 新数据集display_name)."""
    if not state or not state.get("columns"):
        raise ValueError("请先点「1. 读取列信息」。")
    cols = state["columns"]
    mapping = {
        "instruction": _as_list(instruction_cols),
        "input": _as_list(input_cols),
        "think": _as_list(think_cols),
        "output": _as_list(output_cols),
    }
    for role, selected in mapping.items():
        for c in selected:
            if c not in cols:
                raise ValueError(f"{ROLE_LABEL[role]}里有不存在的列 '{c}'，实际列: {cols}。")
    if not mapping["instruction"] and not (fixed_instruction or "").strip():
        raise ValueError("instruction 没有映射任何列，请至少选择 1 列，或在「固定指令」里手写一句。")
    if not mapping["output"]:
        raise ValueError("请至少选择 1 列作为 output 回复。")
    seen: dict[str, str] = {}
    for role, selected in mapping.items():
        for c in selected:
            if c in seen:
                raise ValueError(
                    f"列 '{c}' 同时被映射为{ROLE_LABEL[seen[c]]}和{ROLE_LABEL[role]}，"
                    f"一列只能担任一个角色。")
            seen[c] = role

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
    fixed_ins = (fixed_instruction or "").strip()
    for row in ds:
        out = _merge(row, mapping["output"])
        if not out:
            dropped_empty += 1
            continue
        ins_merged = _merge(row, mapping["instruction"])
        instruction = f"{fixed_ins}\n{ins_merged}" if fixed_ins and ins_merged else (fixed_ins or ins_merged)
        rows.append({
            "instruction": instruction,
            "input": _merge(row, mapping["input"]),
            "think": _merge(row, mapping["think"]),
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
        "mapping": mapping,
        "fixed_instruction": fixed_ins or None,
        "rows_total": len(ds),
        "rows_kept": len(rows),
        "rows_dropped_empty_output": dropped_empty,
        "schema_version": SCHEMA_VERSION,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    use_input = bool(mapping["input"])
    use_think = bool(mapping["think"])
    template, input_columns = build_template(use_input, use_think)
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
    roles_desc = " + ".join(
        f"{role}({len(mapping[role])}列)" for role in ROLES if mapping[role])
    prev_lines = [status, f"映射: {roles_desc}", "-" * 60]
    for i, r in enumerate(rows[:2]):
        filled = template.format(**r)
        if len(filled) > 1000:
            filled = filled[:1000] + "\n…(截断)"
        prev_lines.append(f"[统一后样本 {i+1}]\n{filled}\n" + "-" * 60)
    prev_lines.append("现在可以去「训练」Tab 选中它开始训练（点刷新列表）。")
    return status, "\n".join(prev_lines), name
