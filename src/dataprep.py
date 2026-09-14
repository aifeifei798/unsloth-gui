"""数据处理：把任意来源的数据集制成统一训练数据.

统一 schema（训练只认这个，四个角色都可多选，多列按顺序换行拼接）：
    instruction: str  # 指令/问题（至少映射 1 列）
    input: str        # 补充输入/上下文（可选）
    think: str        # 思维链（可选，仅存档备查）
    output: str       # 回复 = think + output 拼接（至少映射 1 列，
                      #   空回复的行会被丢弃并计数）

模板按实际映射动态组装（没映射 input 就没有 Input 段；Response 恒为 think+output）。
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
from .dataset_utils import drop_empty_sections
from .i18n import t

PROCESSED_ROOT = PROJECT_ROOT / "local_data" / "processed"
UPLOAD_ROOT = PROJECT_ROOT / "local_data" / "uploads"
DATASETS_CONFIG_DIR = PROJECT_ROOT / "datasets_config"

SCHEMA_VERSION = 3

ROLES = ("instruction", "input", "think", "output")


def role_label(role: str) -> str:
    """角色展示名（跟随界面语言）."""
    return {"instruction": t("dp.role.instruction"), "input": t("dp.role.input"),
            "think": t("dp.role.think"),
            "output": t("dp.role.output")}.get(role, role)


def build_template(use_input: bool) -> tuple[str, dict]:
    """按实际映射组装模板与 input_columns（Response 恒为 think+output 合并）。"""
    parts = ["### Instruction:\n{instruction}"]
    columns = {"instruction": "instruction"}
    if use_input:
        parts.append("### Input:\n{input}")
        columns["input"] = "input"
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
            raise ValueError(t("dp.err.no_config"))
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
                raise ValueError(t("dp.err.bad_local", p=p))
        else:
            if existing.is_local:
                raise FileNotFoundError(t("dp.err.local_missing", p=p))
            ds = load_dataset(existing.dataset_id, split=existing.split, streaming=streaming)
    elif kind == "hf":
        hf_id = (hf_id or "").strip()
        if not hf_id:
            raise ValueError(t("dp.err.no_hf"))
        ds = load_dataset(hf_id, split=(split or "train").strip() or "train", streaming=streaming)
    elif kind == "upload":
        if not local_path or not Path(local_path).is_file():
            raise ValueError(t("dp.err.no_upload"))
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
            raise ValueError(t("dp.err.bad_upload", s=p.suffix))
    else:
        raise ValueError(t("dp.err.bad_kind", k=kind))

    if hasattr(ds, "keys") and not hasattr(ds, "map"):  # DatasetDict
        want = (split or "").strip()
        ds = ds[want] if want in ds else ds[list(ds.keys())[0]]
    return ds


def peek_source(kind: str, hf_id: str = "", split: str = "train",
                local_path: str = "", existing: Optional[DatasetConfig] = None,
                ) -> tuple[list, Optional[dict], Optional[int], bool]:
    """只取 1 行做映射预览：远端走 streaming 秒开（不下载全量），本地直接读。
    返回 (列名, 首行|None, 总行数|None, 是否流式)。全量数据等点生成时再拉取。"""
    if kind == "existing" and existing is not None:
        # 已有配置看它自己的切分，和生成侧保持一致
        split = (existing.split or "train").strip() or "train"
    else:
        split = (split or "train").strip() or "train"
    remote = kind == "hf" or (
        kind == "existing" and existing is not None
        and not Path(existing.resolved_dataset_id()).exists()
    )
    if remote:
        from datasets import load_dataset
        ds_id = hf_id.strip() if kind == "hf" else existing.dataset_id
        if not ds_id:
            raise ValueError(t("dp.err.no_hf_id"))
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
    if kind == "existing" and existing is not None:
        split = (existing.split or "train").strip() or "train"
    cols, row, n_rows, streamed = peek_source(kind, hf_id, split, local_path, existing)
    if not cols:
        raise ValueError(t("dp.err.no_columns"))
    count_line = (t("dp.inspect.unknown") if streamed
                  else t("dp.inspect.total", n=n_rows))
    lines = [
        count_line,
        t("dp.inspect.cols", n=len(cols), cols=", ".join(cols)),
        "-" * 60,
    ]
    if row is None:
        lines.append(t("dp.inspect.empty"))
    else:
        preview = {k: (str(v)[:300] + "…" if len(str(v)) > 300 else v) for k, v in row.items()}
        lines.append(t("dp.inspect.sample") + "\n" + json.dumps(preview, ensure_ascii=False, indent=1))
        lines.append("-" * 60)
    state = {"kind": kind, "hf_id": hf_id, "split": split,
             "local_path": local_path,
             "existing_name": existing.display_name if existing else "",
             "columns": cols, "n_rows": n_rows, "streamed": streamed,
             "first_row": row}
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


def resolve_mapping(columns: list, instruction_cols, input_cols,
                      think_cols, output_cols, fixed_instruction: str = "") -> tuple[dict, str]:
    """校验映射并返回 (mapping, 固定指令). 预览和生成共用，保证所见即所得。"""
    mapping = {
        "instruction": _as_list(instruction_cols),
        "input": _as_list(input_cols),
        "think": _as_list(think_cols),
        "output": _as_list(output_cols),
    }
    for role, selected in mapping.items():
        for c in selected:
            if c not in columns:
                raise ValueError(t("dp.err.bad_col", role=role_label(role), c=c,
                                   columns=columns))
    fixed_ins = (fixed_instruction or "").strip()
    if not mapping["instruction"] and not fixed_ins:
        raise ValueError(t("dp.err.no_ins"))
    if not mapping["output"]:
        raise ValueError(t("dp.err.no_out"))
    seen: dict[str, str] = {}
    for role, selected in mapping.items():
        for c in selected:
            if c in seen:
                raise ValueError(t("dp.err.dup_col", c=c, role1=role_label(seen[c]),
                                   role2=role_label(role)))
            seen[c] = role
    return mapping, fixed_ins


def _wrap_think(think: str) -> str:
    """think 已带 <think></think> 就不动，没有就套上，保证 Response 里格式统一."""
    if not think:
        return ""
    low = think.lower()
    if "<think>" in low and "</think>" in low:
        return think
    return f"<think>\n{think}\n</think>"


def format_unified_row(row: dict, mapping: dict, fixed_ins: str = "") -> Optional[dict]:
    """单行转统一格式（含 Response = think + output）。output 为空返回 None（该行丢弃）。"""
    out_raw = _merge(row, mapping["output"])
    if not out_raw:
        return None
    think_raw = _merge(row, mapping["think"])
    think_tagged = _wrap_think(think_raw)
    out = f"{think_tagged}\n{out_raw}" if think_tagged else out_raw
    ins_merged = _merge(row, mapping["instruction"])
    instruction = f"{fixed_ins}\n{ins_merged}" if fixed_ins and ins_merged else (fixed_ins or ins_merged)
    return {
        "instruction": instruction,
        "input": _merge(row, mapping["input"]),
        "think": think_raw,
        "output": out,
    }


def _roles_desc(mapping: dict) -> str:
    desc = " + ".join(t("dp.roles", role=role, n=len(mapping[role]))
                      for role in ROLES if mapping[role])
    if mapping["think"]:
        desc += t("dp.roles.merge")
    return desc


def preview_row(state: dict, instruction_cols, input_cols,
                think_cols, output_cols, fixed_instruction: str = "") -> str:
    """用读取时暂存的首行，按当前映射渲染最终训练文本（与生成逻辑同一套代码）。"""
    if not state or not state.get("columns"):
        raise ValueError(t("dp.err.no_state"))
    row = state.get("first_row")
    if row is None:
        raise ValueError(t("dp.err.empty_source"))
    mapping, fixed_ins = resolve_mapping(
        state["columns"], instruction_cols, input_cols, think_cols, output_cols,
        fixed_instruction)
    unified = format_unified_row(row, mapping, fixed_ins)
    if unified is None:
        return t("dp.preview.empty_out")
    template, _ = build_template(bool(mapping["input"]))
    filled = drop_empty_sections(template.format(**unified))  # 单行预览不截断，完整显示
    roles_desc = _roles_desc(mapping)
    return f"{t('dp.preview.head', roles=roles_desc)}\n" + "-" * 60 + f"\n{filled}"


def generate_unified(state: dict, instruction_cols, input_cols,
                     think_cols, output_cols, output_name: str,
                     fixed_instruction: str = "", progress=None) -> tuple[str, str, str]:
    """生成统一数据。四个角色都可多选；instruction 还支持手写固定文本
   （数据里没有对应列时用它，所有行共用；同时映射了列则做前缀拼在前面）.
    返回 (状态文本, 预览文本, 新数据集display_name)."""
    if not state or not state.get("columns"):
        raise ValueError(t("dp.err.no_state"))
    mapping, fixed_ins = resolve_mapping(
        state["columns"], instruction_cols, input_cols, think_cols, output_cols,
        fixed_instruction)

    name = sanitize_name(output_name)
    cfg_path = DATASETS_CONFIG_DIR / f"{name}.json"
    if cfg_path.exists():
        raise ValueError(t("dp.err.name_exists", name=name, file=cfg_path.name))

    from .config import load_datasets_config, find_by_name
    try:
        if find_by_name(load_datasets_config(), name) is not None:
            raise ValueError(t("dp.err.display_used", name=name))
    except FileNotFoundError:
        pass  # 配置目录还没建，一会儿一起建

    if progress is not None:
        try:
            progress(0.1, desc=t("dp.prog.load"))
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
            progress(0.4, desc=t("dp.prog.map"))
        except Exception:
            pass
    rows: list[dict] = []
    dropped_empty = 0
    for row in ds:
        unified = format_unified_row(row, mapping, fixed_ins)
        if unified is None:
            dropped_empty += 1
            continue
        rows.append(unified)
    if not rows:
        raise ValueError(t("dp.err.no_rows"))

    if progress is not None:
        try:
            progress(0.7, desc=t("dp.prog.save"))
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
    template, input_columns = build_template(use_input)
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
            progress(1.0, desc=t("dp.prog.done"))
        except Exception:
            pass
    status = (t("dp.status.done", name=name, total=len(ds), kept=len(rows))
              + (t("dp.status.dropped", n=dropped_empty) if dropped_empty else ""))
    roles_desc = _roles_desc(mapping)
    prev_lines = [t("dp.mapping", roles=roles_desc), "-" * 60]
    for i, r in enumerate(rows[:2]):
        filled = drop_empty_sections(template.format(**r))
        if len(filled) > 1000:
            filled = filled[:1000] + t("dp.sample.trunc")
        prev_lines.append(f"{t('dp.sample.head', i=i + 1)}\n{filled}\n" + "-" * 60)
    prev_lines.append(t("dp.sample.tail"))
    return status, "\n".join(prev_lines), name
