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

写盘流程：先在 .tmp_* 目录里生成全部产物，最后一步 os.replace 就位，
中途失败不会留下半成品。
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .config import SCHEMA_VERSION, DatasetConfig, find_by_name, load_datasets_config
from .dataset_utils import drop_empty_sections
from .i18n import t
from .paths import DATASETS_CONFIG_DIR, PROCESSED_ROOT, UPLOAD_ROOT
from .storage import atomic_write_json

ROLES = ("instruction", "input", "think", "output")

# 兼容旧代码里的 from .dataprep import SCHEMA_VERSION
__all__ = [
    "SCHEMA_VERSION", "ROLES", "role_label", "build_template", "sanitize_name",
    "load_raw_dataset", "peek_source", "persist_upload", "inspect_text",
    "resolve_mapping", "preview_row", "generate_unified", "format_unified_row",
]


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
    """按来源加载原始数据集（未格式化）. kind: upload | hf | existing."""
    from datasets import load_dataset, load_from_disk

    if kind == "existing":
        if existing is None:
            raise ValueError(t("dp.err.no_config"))
        p = Path(existing.resolved_dataset_id())
        if p.exists():
            if p.is_dir():
                ds = load_from_disk(str(p))  # 本地落盘本来就快，不走流式
            elif p.suffix in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=str(p), split="train",
                                  streaming=streaming)
            elif p.suffix == ".csv":
                ds = load_dataset("csv", data_files=str(p), split="train",
                                  streaming=streaming)
            elif p.suffix in (".parquet",):
                ds = load_dataset("parquet", data_files=str(p), split="train",
                                  streaming=streaming)
            elif p.suffix == ".txt":
                ds = load_dataset("text", data_files=str(p), split="train",
                                  streaming=streaming)
            else:
                raise ValueError(t("dp.err.bad_local", p=p))
        else:
            if existing.is_local:
                raise FileNotFoundError(t("dp.err.local_missing", p=p))
            ds = load_dataset(existing.dataset_id, split=existing.split,
                              streaming=streaming)
    elif kind == "hf":
        hf_id = (hf_id or "").strip()
        if not hf_id:
            raise ValueError(t("dp.err.no_hf"))
        ds = load_dataset(hf_id, split=(split or "train").strip() or "train",
                          streaming=streaming)
    elif kind == "upload":
        if not local_path or not Path(local_path).is_file():
            raise ValueError(t("dp.err.no_upload"))
        p = Path(local_path)
        if p.suffix in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=str(p), split="train",
                              streaming=streaming)
        elif p.suffix == ".csv":
            ds = load_dataset("csv", data_files=str(p), split="train",
                              streaming=streaming)
        elif p.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(p), split="train",
                              streaming=streaming)
        elif p.suffix == ".txt":
            ds = load_dataset("text", data_files=str(p), split="train",
                              streaming=streaming)
        else:
            raise ValueError(t("dp.err.bad_upload", s=p.suffix))
    else:
        raise ValueError(t("dp.err.bad_kind", k=kind))

    if hasattr(ds, "keys") and not hasattr(ds, "map"):  # DatasetDict
        want = (split or "").strip()
        ds = ds[want] if want in ds else ds[list(ds.keys())[0]]
    return ds


def _count_lines(path: Path, limit_bytes: int = 50 * 1024 * 1024) -> Optional[int]:
    """小文件才数行，大文件返回 None，避免预览卡住。"""
    try:
        if path.stat().st_size > limit_bytes:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return None


def _peek_local_file(path: Path) -> tuple[list, Optional[dict], Optional[int]]:
    """只读本地文件的头部，不把整个文件拉进内存。"""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    return [], None, _count_lines(path)
                if isinstance(obj, dict):
                    return list(obj.keys()), obj, _count_lines(path)
                return ["text"], {"text": str(obj)}, _count_lines(path)
        return [], None, 0
    if suffix == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            row = data[0] if data else None
            if isinstance(row, dict):
                return list(row.keys()), row, len(data)
            if row is not None:
                return ["text"], {"text": str(row)}, len(data)
            return [], None, 0
        if isinstance(data, dict):
            return list(data.keys()), data, 1
        return [], None, None
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            row = next(reader, None)
            cols = list(reader.fieldnames or [])
            n = _count_lines(path)
            if n is not None and row is not None:
                n = max(0, n - 1)  # 去掉表头
            return cols, row, n
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(str(path))
            n = int(pf.metadata.num_rows)
            batch = next(pf.iter_batches(batch_size=1), None)
            row = batch.to_pylist()[0] if batch is not None else None
            return (list(row.keys()) if isinstance(row, dict) else []), row, n
        except Exception:
            pass
    if suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().rstrip("\n")
        return ["text"], {"text": first}, _count_lines(path)
    return [], None, None


def peek_source(kind: str, hf_id: str = "", split: str = "train",
                local_path: str = "", existing: Optional[DatasetConfig] = None,
                ) -> tuple[list, Optional[dict], Optional[int], bool]:
    """只取 1 行做映射预览：远端走 streaming 秒开，本地文件只读头部。
    返回 (列名, 首行|None, 总行数|None, 是否流式)。"""
    if kind == "existing" and existing is not None:
        split = (existing.split or "train").strip() or "train"
    else:
        split = (split or "train").strip() or "train"
    remote = kind == "hf" or (
        kind == "existing" and existing is not None
        and not Path(existing.resolved_dataset_id()).exists()
    )
    if remote:
        from datasets import load_dataset
        ds_id = (hf_id or "").strip() if kind == "hf" else existing.dataset_id
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

    local = None
    if kind == "upload":
        local = Path(local_path) if local_path else None
    elif kind == "existing" and existing is not None:
        p = Path(existing.resolved_dataset_id())
        local = p if p.is_file() else None
    if local is not None and local.is_file():
        cols, row, n_rows = _peek_local_file(local)
        if cols:
            return cols, row, n_rows, False

    ds = load_raw_dataset(kind, hf_id, split, local_path, existing)
    cols = list(ds.column_names)
    row = ds[0] if len(ds) > 0 else None
    return cols, row, len(ds), False


def persist_upload(src_path: str, original_name: str | None = None) -> str:
    """把上传文件持久化到 local_data/uploads/，文件名带时间戳 + 短 uuid，避免覆盖。"""
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(t("dp.err.no_upload"))
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve().is_relative_to(UPLOAD_ROOT.resolve()):
            return str(src)
    except Exception:
        pass
    raw_name = original_name or src.name
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", raw_name).strip("._") or "upload"
    safe = safe[:120]
    dst = UPLOAD_ROOT / (
        f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}_{safe}"
    )
    shutil.copy2(src, dst)
    return str(dst)


def _preview_value(v) -> str:
    if isinstance(v, (list, dict)):
        try:
            s = json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            s = str(v)
    else:
        s = str(v)
    return s[:300] + ("…" if len(s) > 300 else "")


def inspect_text(kind: str, hf_id: str = "", split: str = "train",
                 local_path: str = "", existing: Optional[DatasetConfig] = None,
                 ) -> tuple[str, dict]:
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
        preview = {k: _preview_value(v) for k, v in row.items()}
        lines.append(t("dp.inspect.sample") + "\n"
                     + json.dumps(preview, ensure_ascii=False, indent=1))
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
    parts = []
    for c in cols:
        s = _cell(row.get(c))
        if s:
            parts.append(s)
    return "\n".join(parts)


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
    instruction = (f"{fixed_ins}\n{ins_merged}" if fixed_ins and ins_merged
                   else (fixed_ins or ins_merged))
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
    try:
        filled = drop_empty_sections(template.format(**unified))
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(t("cfg.ds_tpl_bad", err=e)) from e
    roles_desc = _roles_desc(mapping)
    return f"{t('dp.preview.head', roles=roles_desc)}\n" + "-" * 60 + f"\n{filled}"


def _config_exists(name: str) -> bool:
    if (DATASETS_CONFIG_DIR / f"{name}.json").exists():
        return True
    try:
        return find_by_name(load_datasets_config(warnings=[]), name) is not None
    except FileNotFoundError:
        return False


def generate_unified(state: dict, instruction_cols, input_cols,
                     think_cols, output_cols, output_name: str,
                     fixed_instruction: str = "", progress=None,
                     cancel_event=None) -> tuple[str, str, str]:
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
    out_dir = PROCESSED_ROOT / name
    if cfg_path.exists():
        raise ValueError(t("dp.err.name_exists", name=name, file=cfg_path.name))
    if _config_exists(name):
        raise ValueError(t("dp.err.display_used", name=name))
    if out_dir.exists():
        raise ValueError(t("dp.err.dir_exists", name=name))

    existing = None
    if state.get("kind") == "existing" and state.get("existing_name"):
        try:
            existing = find_by_name(load_datasets_config(warnings=[]),
                                    state["existing_name"])
        except Exception:
            existing = None

    _progress(progress, 0.1, t("dp.prog.load"))
    ds = load_raw_dataset(state["kind"], state.get("hf_id", ""),
                          state.get("split", "train"),
                          state.get("local_path", ""), existing)
    total = None
    try:
        total = len(ds)
    except Exception:
        total = None

    _progress(progress, 0.4, t("dp.prog.map"))
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = PROCESSED_ROOT / f".tmp_{name}_{uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    cfg_tmp = DATASETS_CONFIG_DIR / f".{name}.{uuid4().hex[:8]}.tmp.json"
    samples: list[dict] = []
    kept = 0
    dropped = 0
    step = max(1, (total // 20) if total else 500)
    try:
        data_path = tmp_dir / "data.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds, 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError(t("dp.err.cancelled"))
                unified = format_unified_row(row, mapping, fixed_ins)
                if unified is None:
                    dropped += 1
                    continue
                f.write(json.dumps(unified, ensure_ascii=False) + "\n")
                kept += 1
                if len(samples) < 2:
                    samples.append(unified)
                if total and i % step == 0:
                    frac = 0.4 + 0.3 * (i / max(total, 1))
                    _progress(progress, min(frac, 0.7),
                              t("dp.prog.map_n", i=i, total=total))
        if kept == 0:
            raise ValueError(t("dp.err.no_rows"))

        _progress(progress, 0.7, t("dp.prog.save"))
        from datasets import load_dataset

        hf_ds = load_dataset("json", data_files=str(data_path), split="train")
        hf_ds.save_to_disk(str(tmp_dir / "hf_dataset"))

        manifest = {
            "name": name,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": {k: state.get(k) for k in
                       ("kind", "hf_id", "split", "existing_name", "local_path")},
            "mapping": mapping,
            "fixed_instruction": fixed_ins or None,
            "rows_total": total if total is not None else kept + dropped,
            "rows_kept": kept,
            "rows_dropped_empty_output": dropped,
            "schema_version": SCHEMA_VERSION,
        }
        atomic_write_json(tmp_dir / "manifest.json", manifest)

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
        atomic_write_json(cfg_tmp, cfg)

        # 最后一步才就位：先搬目录，再落配置；配置失败则回滚目录
        os.replace(tmp_dir, out_dir)
        try:
            os.replace(cfg_tmp, cfg_path)
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise

        _progress(progress, 1.0, t("dp.prog.done"))
        status = (t("dp.status.done", name=name, total=manifest["rows_total"],
                    kept=kept)
                  + (t("dp.status.dropped", n=dropped) if dropped else ""))
        roles_desc = _roles_desc(mapping)
        prev_lines = [t("dp.mapping", roles=roles_desc), "-" * 60]
        for i, r in enumerate(samples):
            try:
                filled = drop_empty_sections(template.format(**r))
            except (KeyError, IndexError, ValueError):
                filled = str(r)
            if len(filled) > 1000:
                filled = filled[:1000] + t("dp.sample.trunc")
            prev_lines.append(f"{t('dp.sample.head', i=i + 1)}\n{filled}\n"
                              + "-" * 60)
        prev_lines.append(t("dp.sample.tail"))
        return status, "\n".join(prev_lines), name
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if cfg_tmp.exists():
            try:
                cfg_tmp.unlink()
            except OSError:
                pass


def _progress(progress, frac: float, desc: str) -> None:
    if progress is None:
        return
    try:
        progress(frac, desc=desc)
    except Exception:
        pass
