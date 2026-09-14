"""数据集准备：修复原来 prompt_template.format 脆弱、schema 不一致就炸等问题."""
from __future__ import annotations

import string
from pathlib import Path
from typing import Optional

from .config import DatasetConfig


def validate_template_columns(cfg: DatasetConfig) -> list[str]:
    """检查 prompt_template 里的 {占位符} 是否都能在 input_columns.key 里找到."""
    fields = [fn for _, fn, _, _ in string.Formatter().parse(cfg.prompt_template) if fn]
    missing = [f for f in fields if f not in cfg.input_columns]
    return missing


def _format_row(prompt_template: str, column_mappings: dict, row: dict, idx: int) -> str:
    fmt: dict = {}
    for placeholder, col in column_mappings.items():
        if col not in row:
            raise ValueError(
                f"第 {idx} 行缺少列 '{col}'（input_columns 映射 '{placeholder}' -> '{col}'）。"
                f"该数据集实际列: {sorted(row.keys())}。请检查 datasets_config 映射。"
            )
        v = row[col]
        fmt[placeholder] = "" if v is None else str(v)
    try:
        return prompt_template.format(**fmt)
    except KeyError as e:
        raise ValueError(
            f"prompt_template 里的占位符 {e} 在 input_columns Lima没有对应 key。"
            f"template 需要: {sorted(fmt.keys())}。请检查 JSON 配置。"
        ) from e


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
        raise ValueError(
            f"数据集 '{cfg.display_name}' 的 prompt_template 占位符 {missing} "
            f"在 input_columns {sorted(cfg.input_columns.keys())} 中找不到对应。"
        )

    p = Path(ds_path)
    if p.exists():
        if p.is_dir():
            try:
                dataset = load_from_disk(ds_path)
            except Exception as e:
                raise RuntimeError(f"本地数据集加载失败 {ds_path}: {e}") from e
        elif p.suffix in (".jsonl", ".json"):
            dataset = load_dataset("json", data_files=ds_path, split="train")
        elif p.suffix == ".csv":
            dataset = load_dataset("csv", data_files=ds_path, split="train")
        elif p.suffix == ".txt":
            dataset = load_dataset("text", data_files=ds_path, split="train")
        else:
            raise ValueError(f"不支持的本地数据格式: {ds_path}（支持目录/jsonl/json/csv/txt）")
    else:
        if cfg.is_local:
            raise FileNotFoundError(f"本地数据集路径未找到: {ds_path}")
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
                raise ValueError(
                    f"数据集 '{cfg.display_name}' 缺少列 '{c}'，实际列: {sorted(examples.keys())}。"
                )
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
        raise ValueError("没有可合并的数据集。")
    if len(datasets) == 1:
        ds = datasets[0]
        keep = [c for c in ["text"] if c in ds.column_names]
        return ds.remove_columns([c for c in ds.column_names if c not in keep]) if keep else ds
    aligned = []
    for ds in datasets:
        if "text" not in ds.column_names:
            raise ValueError(f"某个数据集缺少 text 列，实际列: {ds.column_names}")
        aligned.append(ds.remove_columns([c for c in ds.column_names if c != "text"]))
    return concatenate_datasets(aligned)


def dataset_preview_text(cfg: DatasetConfig, n: int = 3, max_chars: int = 1200) -> str:
    """UI 预览用：只读前 n 条格式化结果 + 列信息，不触发完整 map 全量开销过大时自动截断."""
    try:
        ds = prepare_dataset(cfg, truncate_for_testing=True, max_samples=max(20, n))
    except Exception as e:
        return f"❌ 预览失败: {e}"
    lines = [
        f"数据集: {cfg.display_name}",
        f"来源: {cfg.resolved_dataset_id()} | 切分: {cfg.split}",
        f"原始列: {ds.column_names} | 总条数(已截断预览): {len(ds)}",
        "-" * 60,
    ]
    for i in range(min(n, len(ds))):
        t = ds[i]["text"]
        if len(t) > max_chars:
            t = t[:max_chars] + f"\n…(截断，共 {len(ds[i]['text'])} 字符)"
        lines.append(f"[样本 {i+1}]\n{t}\n" + "-" * 60)
    bad = validate_template_columns(cfg)
    if bad:
        lines.append(f"⚠️ template 占位符缺映射: {bad}")
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
