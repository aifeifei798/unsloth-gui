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
    except (IndexError, ValueError) as e:
        raise ValueError(t("cfg.ds_tpl_bad", err=e)) from e


def load_raw_dataset(
    cfg: DatasetConfig,
    truncate_for_testing: bool = False,
    max_samples: int = 200,
    preview_rows: int | None = None,
):
    """只加载原始数据（不套模板）。

    训练要先拿到模型 tokenizer 才能决定用哪种格式渲染，所以拆成
    「加载」和「格式化」两步；预览仍走 prepare_dataset 一步到位。

    - 本地：优先 load_from_disk；若是 .jsonl/.json/.csv/.txt 文件则自动转换
    - 远端：load_dataset(dataset_id, split)
    """
    from datasets import load_dataset, load_from_disk

    ds_path = cfg.resolved_dataset_id()
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

    if preview_rows is not None and len(dataset) > int(preview_rows):
        dataset = dataset.select(range(int(preview_rows)))
    if truncate_for_testing and len(dataset) > int(max_samples):
        dataset = dataset.select(range(int(max_samples)))
    return dataset


def format_with_template(dataset, cfg: DatasetConfig):
    """套 cfg.prompt_template 渲染成 text 列（老模式：### Instruction / ### Response）。"""
    missing = validate_template_columns(cfg)
    if missing:
        raise ValueError(t("cfg.tpl_missing", name=cfg.display_name, missing=missing,
                           keys=sorted(cfg.input_columns.keys())))
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

    return dataset.map(formatting_prompts_func, batched=True, remove_columns=None)


def prepare_dataset(
    cfg: DatasetConfig,
    truncate_for_testing: bool = False,
    max_samples: int = 200,
    preview_only: bool = False,
    preview_rows: int | None = None,
):
    """加载 + 用 prompt_template 格式化为带 text 列的数据集（预览 / 老模式训练）。"""
    dataset = load_raw_dataset(cfg, truncate_for_testing, max_samples, preview_rows)
    return format_with_template(dataset, cfg)


# ---------------------------------------------------------------- chat 模板模式

def has_chat_template(tokenizer) -> bool:
    """tokenizer 是否带可用的对话模板（模型自带的格式）。"""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return False
    try:
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        return False


def supports_chat_roles(cfg: DatasetConfig) -> bool:
    """数据集映射里是否齐了 chat 模式需要的角色（instruction / output）。"""
    keys = set((cfg.input_columns or {}).keys())
    return "instruction" in keys and "output" in keys


def render_chat_text(tokenizer, messages, add_generation_prompt: bool = False,
                     enable_thinking=None) -> str:
    """用模型自带 chat 模板把 messages 渲染成纯文本。"""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(t("cfg.no_chat_template"))
    kwargs: dict = dict(tokenize=False, add_generation_prompt=bool(add_generation_prompt))
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        if "enable_thinking" in kwargs:
            kwargs.pop("enable_thinking")
            try:
                text = tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as e2:
                raise ValueError(t("cfg.chat_tpl_fail", err=e2)) from e2
        else:
            raise ValueError(t("cfg.chat_tpl_fail", err=e)) from e
    text = "" if text is None else str(text)
    if not text.strip():
        raise ValueError(t("cfg.chat_tpl_empty"))
    return text


def chat_messages_from_row(row: dict, column_mappings: dict, idx: int = 0,
                           name: str = "") -> list[dict]:
    """统一数据的一行 → chat messages：user=指令(+上下文)，assistant=回复(含思考)。"""
    def pick(role: str) -> str:
        col = (column_mappings or {}).get(role)
        if not col:
            return ""
        if col not in row:
            raise ValueError(t("cfg.ds_col_missing", i=idx, col=col, ph=role,
                               actual=sorted(row.keys())))
        v = row[col]
        return "" if v is None else str(v).strip()

    instruction = pick("instruction")
    context = pick("input")
    answer = pick("output")
    if not answer:
        raise ValueError(t("cfg.ds_empty_output", name=name, i=idx))
    user = instruction
    if context:
        user = (user + "\n\n" + context) if user else context
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": answer}]


def format_with_chat_template(dataset, cfg: DatasetConfig, tokenizer):
    """用模型自带 chat 模板渲染成 text 列（新模式，训练与推理格式一致）。"""
    column_mappings = cfg.input_columns
    if not column_mappings:
        raise ValueError(t("cfg.ds_bad_columns", name=cfg.display_name))

    def formatting_func(examples):
        cols = list(column_mappings.values())
        for c in cols:
            if c not in examples:
                raise ValueError(t("cfg.ds_need_col", name=cfg.display_name, col=c,
                                   actual=sorted(examples.keys())))
        texts = []
        for i in range(len(examples[cols[0]])):
            row = {c: examples[c][i] for c in cols}
            messages = chat_messages_from_row(row, column_mappings, i, cfg.display_name)
            texts.append(render_chat_text(tokenizer, messages))
        return {"text": texts}

    return dataset.map(formatting_func, batched=True, remove_columns=None)


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
        ds = prepare_dataset(cfg, preview_rows=max(20, n))
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
