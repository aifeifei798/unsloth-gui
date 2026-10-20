"""配置加载与校验。

设计目标：
- 向后兼容现有 models.json / datasets_config/*.json
- 路径一律相对项目根目录解析，支持 ~，不依赖启动 cwd
- models.json 严格校验（写回时不会丢条目）
- datasets_config 逐文件容错：坏文件只跳过并返回告警，不拖垮整个列表
- display_name 唯一性校验，缺字段时给出可读错误而不是 KeyError
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .i18n import t
from .paths import (
    DATASETS_CONFIG_DIR,
    MODELS_CONFIG,
    PROJECT_ROOT,
    resolve_local,
)
from .storage import atomic_write_json

# 统一数据 schema 版本；数据处理生成时写入，训练前强校验
SCHEMA_VERSION = 3


@dataclass
class ModelConfig:
    display_name: str
    model_id: str
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # "bfloat16" | "float16" | None(auto)
    max_seq_length: int = 2048
    chat_template: Optional[str] = None  # 如 "llama-3.1" / "qwen-2.5" / "gemma-3"，None 则不强制
    source: str = "auto"  # auto | local | huggingface | modelscope
    extra: dict = field(default_factory=dict)

    def resolved_model_id(self) -> str:
        """相对路径（如 ../xxx / ./xxx / ~/xxx）相对项目根解析，其余按 HF id 原样返回."""
        mid = (self.model_id or "").strip()
        if not mid:
            return ""
        if mid.startswith(("~", "/", "./", "../")) or mid in (".", ".."):
            return str(resolve_local(mid))
        maybe_local = PROJECT_ROOT / mid
        if maybe_local.exists():
            return str(maybe_local.resolve())
        return mid


@dataclass
class DatasetConfig:
    display_name: str
    dataset_id: str
    split: str = "train"
    is_local: bool = False
    prompt_template: str = ""
    input_columns: dict = field(default_factory=dict)
    chat_template: Optional[str] = None
    recommended_params: dict = field(default_factory=dict)
    # 数据处理 Tab 制成的统一数据才带此标记；训练只认 processed=True + schema 匹配的
    processed: bool = False
    schema_version: int = 0
    source_file: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def resolved_dataset_id(self) -> str:
        did = (self.dataset_id or "").strip()
        if not did:
            return ""
        if self.is_local or did.startswith(("~", "/", "./", "../")) or did in (".", ".."):
            return str(resolve_local(did))
        # 兼容：即使没标 is_local，只要本地存在就按本地处理
        maybe_local = PROJECT_ROOT / did
        if maybe_local.exists():
            return str(maybe_local.resolve())
        return did


def _ensure_unique_display_names(items: list[dict], source: str) -> None:
    seen: dict[str, str] = {}
    for it in items:
        name = it.get("display_name")
        if not name:
            raise ValueError(t("cfg.no_display", src=source, item=it))
        if name in seen:
            raise ValueError(t("cfg.dup", src=source, name=name))
        seen[name] = source


def _int_field(value: Any, default: int, *, name: str, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            t("cfg.bad_int", name=name, field=field_name, value=value)
        ) from e


def _model_from_dict(d: dict) -> ModelConfig:
    if "display_name" not in d or "model_id" not in d:
        raise ValueError(t("cfg.model_missing", d=d))
    known = {"display_name", "model_id", "load_in_4bit", "dtype", "max_seq_length",
             "chat_template", "source"}
    source = str(d.get("source", "auto") or "auto").lower()
    if source not in ("auto", "local", "huggingface", "modelscope"):
        raise ValueError(t("cfg.model_bad_source", name=d.get("display_name"),
                           source=source))
    return ModelConfig(
        display_name=str(d["display_name"]).strip(),
        model_id=str(d["model_id"]).strip(),
        load_in_4bit=bool(d.get("load_in_4bit", True)),
        dtype=d.get("dtype"),
        max_seq_length=_int_field(d.get("max_seq_length"), 2048,
                                  name=d.get("display_name"), field_name="max_seq_length"),
        chat_template=d.get("chat_template"),
        source=source,
        extra={k: v for k, v in d.items() if k not in known},
    )


def _dataset_from_dict(d: dict, source_file: Optional[str] = None) -> DatasetConfig:
    if "display_name" not in d or "dataset_id" not in d:
        raise ValueError(t("cfg.ds_missing", d=d))
    if not d.get("prompt_template"):
        raise ValueError(t("cfg.ds_no_template", name=d.get("display_name")))
    if not isinstance(d.get("input_columns"), dict) or not d["input_columns"]:
        raise ValueError(t("cfg.ds_bad_columns", name=d.get("display_name")))
    known = {"display_name", "dataset_id", "split", "is_local", "prompt_template",
             "input_columns", "chat_template", "recommended_params",
             "processed", "schema_version", "_source_file"}
    return DatasetConfig(
        display_name=str(d["display_name"]).strip(),
        dataset_id=str(d["dataset_id"]),
        split=d.get("split", "train"),
        is_local=bool(d.get("is_local", False)),
        prompt_template=d["prompt_template"],
        input_columns=dict(d["input_columns"]),
        chat_template=d.get("chat_template"),
        recommended_params=dict(d.get("recommended_params") or {}),
        processed=bool(d.get("processed", False)),
        schema_version=_int_field(d.get("schema_version"), 0,
                                 name=d.get("display_name"), field_name="schema_version"),
        source_file=source_file or d.get("_source_file"),
        extra={k: v for k, v in d.items() if k not in known and k != "use_full_dataset"},
    )


def load_models_config(path: Path | str = MODELS_CONFIG) -> list[ModelConfig]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(t("cfg.model_not_found", path=path))
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # 兼容单对象写法
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError(t("cfg.model_not_list", path=path))
    _ensure_unique_display_names(raw, str(path))
    return [_model_from_dict(d) for d in raw]


def load_datasets_config(path: Path | str = DATASETS_CONFIG_DIR,
                         warnings: list[str] | None = None) -> list[DatasetConfig]:
    """加载数据集配置。

    单个文件损坏/缺字段/重名时跳过该文件并往 `warnings` 里追加说明，
    其余文件照常返回。只有「路径不存在」才抛异常；空目录返回空列表。
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raws: list[tuple[dict, str]] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            # 空目录是合法状态（用户可能刚删完最后一份），返回空列表
            return []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                _warn(warnings, t("warn.ds_file_bad", file=fp.name, err=e))
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    _warn(warnings, t("warn.ds_entry_bad", file=fp.name))
                    continue
                raws.append((item, fp.name))
    elif path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                raws.append((item, path.name))
            else:
                _warn(warnings, t("warn.ds_entry_bad", file=path.name))
    else:
        raise FileNotFoundError(t("cfg.ds_not_found", path=path))

    configs: list[DatasetConfig] = []
    seen: dict[str, str] = {}
    for raw, file_name in raws:
        name = str(raw.get("display_name") or "").strip()
        if name and name in seen:
            _warn(warnings, t("warn.ds_duplicate", name=name, file=file_name,
                              first=seen[name]))
            continue
        try:
            cfg = _dataset_from_dict(raw, source_file=file_name)
        except Exception as e:
            _warn(warnings, t("warn.ds_entry_bad", file=file_name, err=e))
            continue
        if not cfg.display_name:
            _warn(warnings, t("warn.ds_entry_bad", file=file_name,
                              err=t("cfg.no_display", src=file_name, item=raw)))
            continue
        seen[cfg.display_name] = file_name
        configs.append(cfg)
    return configs


def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(message)


def safe_load_configs() -> tuple[list[ModelConfig], list[DatasetConfig], list[str]]:
    """UI 启动时调用：失败也不抛异常，返回 warnings 供界面展示."""
    warnings: list[str] = []
    try:
        models = load_models_config()
    except Exception as e:
        warnings.append(t("warn.models", err=e))
        models = []
    try:
        datasets = load_datasets_config(warnings=warnings)
    except Exception as e:
        warnings.append(t("warn.datasets", err=e))
        datasets = []
    return models, datasets, warnings


def find_by_name(items: list, name: str) -> Optional[Any]:
    for it in items:
        if getattr(it, "display_name", None) == name:
            return it
    return None


def source_label(code: str) -> str:
    """来源展示名（跟随界面语言），存文件永远用 code。"""
    return {"local": t("model.src.local"), "huggingface": t("model.src.hf"),
            "modelscope": t("model.src.ms")}.get((code or "").lower(), code or "")


def model_source(cfg: ModelConfig) -> str:
    """判定模型来源：显式配置优先，auto 则看本地是否存在该路径."""
    s = (cfg.source or "auto").lower()
    if s in ("local", "huggingface", "modelscope"):
        return s
    mid = (cfg.model_id or "").strip()
    if mid.startswith(("~", "/", "./", "../")) or mid in (".", ".."):
        return "local"
    if (PROJECT_ROOT / mid).exists():
        return "local"
    return "huggingface"


def is_trainable_dataset(cfg: DatasetConfig) -> bool:
    """训练只认：processed + 当前 schema + 产物真实存在。"""
    if not cfg.processed:
        return False
    if int(cfg.schema_version or 0) != SCHEMA_VERSION:
        return False
    try:
        return Path(cfg.resolved_dataset_id()).exists()
    except Exception:
        return False


def trainable_reason(cfg: DatasetConfig) -> str:
    """给 UI 用的「为什么不能训练」，可训练时返回空串。"""
    if not cfg.processed:
        return t("cfg.not_processed")
    if int(cfg.schema_version or 0) != SCHEMA_VERSION:
        return t("cfg.schema_old", version=cfg.schema_version, current=SCHEMA_VERSION)
    p = Path(cfg.resolved_dataset_id())
    if not p.exists():
        return t("cfg.ds_path_missing", path=p)
    return ""


def ensure_local_model(cfg: ModelConfig) -> str:
    """返回可直接喂给 FastLanguageModel.from_pretrained 的地址.

    - 本地：解析后的绝对路径（不存在则报错）
    - HuggingFace：ID 原样返回（首次使用自动下载）
    - 魔搭：snapshot_download 落到本地缓存后返回目录（需 pip install modelscope）
    """
    src = model_source(cfg)
    mid = (cfg.model_id or "").strip()
    if not mid:
        raise ValueError(t("cfg.model_empty", name=cfg.display_name))
    if src == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as e:
            raise RuntimeError(t("cfg.modelscope_need")) from e
        return snapshot_download(mid)
    if src == "local":
        p = cfg.resolved_model_id()
        if not Path(p).exists():
            raise FileNotFoundError(t("cfg.model_local_missing", id=cfg.model_id, path=p))
        return p
    return mid


def model_to_dict(cfg: ModelConfig) -> dict:
    d: dict = {
        "display_name": cfg.display_name,
        "model_id": cfg.model_id,
        "load_in_4bit": bool(cfg.load_in_4bit),
        "dtype": cfg.dtype,
    }
    if cfg.max_seq_length != 2048:
        d["max_seq_length"] = int(cfg.max_seq_length)
    if cfg.chat_template:
        d["chat_template"] = cfg.chat_template
    if (cfg.source or "auto") != "auto":
        d["source"] = cfg.source
    d.update(cfg.extra or {})
    return d


def save_models_config(models: list[ModelConfig],
                       path: Path | str = MODELS_CONFIG) -> None:
    """原子写回 models.json（先写临时文件再替换，防写一半崩）。"""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    atomic_write_json(path, [model_to_dict(m) for m in models])
