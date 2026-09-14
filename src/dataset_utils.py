"""数据集准备：修复原来 prompt_template.format 脆弱、schema 不一致就炸等问题."""
from __future__ import annotations

import re
import string
from pathlib import Path

from .config import DatasetConfig
from .i18n import t


def validate_template_columns(cfg: DatasetConfig) -> list[str]:
    """检查 prompt_template 里的 {占位符} 是否都能在 input_columns.key 里找到."""
    fields = [fn for _, fn, _, _ in string.Formatter().parse(cfg.prompt_template) if fn]
    missing = [f for f in fields if f not in cfg.input_columns]
    return missing


# 内容为空的段落头（如某行没有 input 上下文时的 "### Input:" 空段），
# 后面只跟空白 + 下一个段落或结尾时整段删掉，避免浪费 token。
# 只认我们自己的段头，不碰正文内容；Response 恒非空，不在清理之列。
_EMPTY_SECTION_RE = re.compile(
    r"### (?:Instruction|Input|Thinking):\n(?:[ \t]*\n)+(?=### |\s*\Z)"
)


def drop_empty_sections(text: str) -> str:
    """删掉内容为空的模板段落。生成侧预览共用，保证所见即所得。"""
    return _EMPTY_SECTION_RE.sub("", text)


def _format_row(prompt_template: str, column_mappings: dict, row: dict, idx: int) -> str:
    fmt: dict = {}
    for placeholder, col in column_mappings.items():
        if col not in row:
            raise ValueError(t("cfg.ds_col_missing", i=idx, col=col, ph=placeholder,
                               actual=sorted(row.keys())))
        v = row[col]
        fmt[placeholder] = "" if v is None else str(v)
    try:
        return drop_empty_sections(prompt_template.format(**fmt))
    except KeyError as e:
        raise ValueError(t("cfg.ds_tpl_key", key=e, need=sorted(fmt.keys()))) from e


def prepare_dataset(
    cfg: DatasetConfig,
    truncate_for_testing: bool = False,
    max_samples: int = 200,
    preview_only: bool = False,
):
    """加载 + 格式化为带 text 列的数据集.

    - 本地：优先 load_from_disk；若是 .jsonl/.json 文件则自动转换（替代原来必须手动跑 convert_data.py）
    - 远端：load_dataset(dataset_id, split)
    - 返回 HF datasets.Dataset（带 text 列）
    """
    from datasets import load_dataset, load_from_disk

    ds_path = cfg.resolved_dataset_id()
    missing = validate_template_columns(cfg)
    if missing:
        raise ValueError(t("cfg.tpl_missing", name=cfg.display_name, missing=missing,
                           keys=sorted(cfg.input_columns.keys())))

    p = Path(ds_path)
    if p.exists():
        if p.is_dir():
            try:
                dataset = load_from_disk(ds_path)
            except Exception as e:
                raise RuntimeError(t("dp.err.disk_fail", path=ds_path, err=e)) from e
        elif p.suffix in (".jsonl", ".json"):
            dataset = load_dataset("json", data_files=ds_path, split="train")
        elif p.suffix == ".csv":
            dataset = load_dataset("csv", data_files=ds_path, split="train")
        elif p.suffix == ".txt":
            dataset = load_dataset("text", data_files=ds_path, split="train")
        else:
            raise ValueError(t("cfg.ds_bad_format", path=p))
    else:
        if cfg.is_local:
            raise FileNotFoundError(t("cfg.ds_path_missing", path=ds_path))
        dataset = load_dataset(cfg.dataset_id, split=cfg.split)

    # 兼容 DatasetDict（load_from_disk 可能存的是切分 dict）
    if hasattr(dataset, "keys") and not hasattr(dataset, "map"):
        # DatasetDict：优先取 cfg.split，否则取第一个切分
        dataset = dataset[cfg.split] if cfg.split in dataset else dataset[list(dataset.keys())[0]]

    prompt_template = cfg.prompt_template
    column_mappings = cfg.input_columns

    def formatting_prompts_func(examples):
        # examples 是 batched dict: {col: [values]}
        cols = list(column_mappings.values())
        for c in cols:
            if c not in examples:
                raise ValueError(t("cfg.ds_need_col", name=cfg.display_name, col=c,
                                   actual=sorted(examples.keys())))
        texts = []
        for i in range(len(examples[cols[0]])):
            row = {c: examples[c][i] for c in cols}
            texts.append(_format_row(prompt_template, column_mappings, row, i))
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=None)

    if truncate_for_testing and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    return dataset


def combine_datasets(datasets: list):
    """合并多个已格式化的 text 数据集：只保留 text 列，避免 schema 不一致炸掉."""
    from datasets import concatenate_datasets

    if not datasets:
        raise ValueError(t("cfg.ds_empty_combine"))
    if len(datasets) == 1:
        ds = datasets[0]
        keep = [c for c in ["text"] if c in ds.column_names]
        return ds.remove_columns([c for c in ds.column_names if c not in keep]) if keep else ds
    aligned = []
    for ds in datasets:
        if "text" not in ds.column_names:
            raise ValueError(t("cfg.ds_no_text", cols=ds.column_names))
        aligned.append(ds.remove_columns([c for c in ds.column_names if c != "text"]))
    return concatenate_datasets(aligned)


def dataset_preview_text(cfg: DatasetConfig, n: int = 3, max_chars: int = 1200) -> str:
    """UI 预览用：只读前 n 条格式化结果 + 列信息，不触发完整 map 全量开销过大时自动截断."""
    try:
        ds = prepare_dataset(cfg, truncate_for_testing=True, max_samples=max(20, n))
    except Exception as e:
        return t("cfg.ds_preview_fail", err=e)
    lines = [
        t("cfg.ds_preview_head", name=cfg.display_name, src=cfg.resolved_dataset_id(),
          split=cfg.split, cols=ds.column_names, n=len(ds)),
    ]
    for i in range(min(n, len(ds))):
        text = ds[i]["text"]
        if len(text) > max_chars:
            text = text[:max_chars] + t("cfg.ds_preview_trunc", n=len(ds[i]["text"]))
        lines.append(t("cfg.ds_preview_sample", i=i + 1, text=text))
    bad = validate_template_columns(cfg)
    if bad:
        lines.append(t("cfg.ds_preview_tpl_warn", missing=bad))
    return "\n".join(lines)


def apply_chat_template_if_needed(tokenizer, cfg: DatasetConfig):
    """如果配置了 chat_template，统一走 unsloth 的 get_chat_template，避免各模型格式手写错."""
    if not cfg.chat_template:
        return tokenizer
    try:
        from unsloth.chat_templates import get_chat_template
    except Exception:
        return tokenizer
    try:
        return get_chat_template(tokenizer, chat_template=cfg.chat_template)
    except Exception:
        return tokenizer
