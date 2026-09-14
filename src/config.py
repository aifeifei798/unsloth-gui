"""配置加载与校验（替代原来 app.py 里裸 json.load 的脆弱逻辑）.

设计目标：
- 向后兼容现有 models.json / datasets_config/*.json
- 路径一律相对项目根目录解析，不依赖启动 cwd
- display_name 唯一性校验，缺字段时给出可读错误而不是 KeyError
- 支持新可选字段：max_seq_length / chat_template / quantization 等
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        """相对路径（如 ../xxx / ./xxx）相对项目根解析，其余按 HF id 原样返回."""
        mid = self.model_id
        if mid.startswith((".", "/", "~")) or "/" not in mid and Path(mid).exists():
            p = (PROJECT_ROOT / mid).expanduser().resolve() if not Path(mid).is_absolute() else Path(mid)
            return str(p)
        # 形如 ../foo 的本地路径（不以 ./ 开头）也做解析
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
    # 数据处理 Tab 制成的统一数据才带此标记；训练只认 processed=True 的
    processed: bool = False
    schema_version: int = 0
    extra: dict = field(default_factory=dict)

    def resolved_dataset_id(self) -> str:
        did = self.dataset_id
        p = Path(did)
        if self.is_local or did.startswith((".", "/", "~")):
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.expanduser().resolve())
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
            raise ValueError(f"{source} 中存在缺少 display_name 的配置: {it}")
        if name in seen:
            raise ValueError(f"{source} 中 display_name 重复: '{name}'，请改名后重试。")
        seen[name] = source


def _model_from_dict(d: dict) -> ModelConfig:
    if "display_name" not in d or "model_id" not in d:
        raise ValueError(f"模型配置缺字段（需要 display_name/model_id）: {d}")
    known = {"display_name", "model_id", "load_in_4bit", "dtype", "max_seq_length",
             "chat_template", "source"}
    source = str(d.get("source", "auto") or "auto").lower()
    if source not in ("auto", "local", "huggingface", "modelscope"):
        raise ValueError(f"模型 '{d.get('display_name')}' 的 source 非法: {source}，"
                         f"只能是 auto/local/huggingface/modelscope。")
    return ModelConfig(
        display_name=d["display_name"],
        model_id=d["model_id"],
        load_in_4bit=bool(d.get("load_in_4bit", True)),
        dtype=d.get("dtype"),
        max_seq_length=int(d.get("max_seq_length", 2048)),
        chat_template=d.get("chat_template"),
        source=source,
        extra={k: v for k, v in d.items() if k not in known},
    )


def _dataset_from_dict(d: dict) -> DatasetConfig:
    if "display_name" not in d or "dataset_id" not in d:
        raise ValueError(f"数据集配置缺字段（需要 display_name/dataset_id）: {d}")
    if not d.get("prompt_template"):
        raise ValueError(f"数据集 '{d.get('display_name')}' 缺少 prompt_template。")
    if not isinstance(d.get("input_columns"), dict) or not d["input_columns"]:
        raise ValueError(f"数据集 '{d.get('display_name')}' 的 input_columns 必须是非空 dict。")
    known = {"display_name", "dataset_id", "split", "is_local", "prompt_template",
             "input_columns", "chat_template", "recommended_params",
             "processed", "schema_version"}
    return DatasetConfig(
        display_name=d["display_name"],
        dataset_id=d["dataset_id"],
        split=d.get("split", "train"),
        is_local=bool(d.get("is_local", False)),
        prompt_template=d["prompt_template"],
        input_columns=dict(d["input_columns"]),
        chat_template=d.get("chat_template"),
        recommended_params=dict(d.get("recommended_params") or {}),
        processed=bool(d.get("processed", False)),
        schema_version=int(d.get("schema_version", 0)),
        extra={k: v for k, v in d.items() if k not in known and k != "use_full_dataset"},
    )


def load_models_config(path: Path | str = PROJECT_ROOT / "models.json") -> list[ModelConfig]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"模型配置文件未找到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # 兼容单对象写法
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} 必须是非空 list。")
    _ensure_unique_display_names(raw, str(path))
    return [_model_from_dict(d) for d in raw]


def load_datasets_config(path: Path | str = PROJECT_ROOT / "datasets_config") -> list[DatasetConfig]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raws: list[dict] = []
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"数据集配置目录为空: {path}")
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["_source_file"] = fp.name
            raws.append(d)
    elif path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raws = data if isinstance(data, list) else [data]
    else:
        raise FileNotFoundError(f"数据集配置未找到: {path}")
    # 去掉内部字段后再校验重名
    for d in raws:
        d.pop("_source_file", None)
    _ensure_unique_display_names(raws, str(path))
    return [_dataset_from_dict(d) for d in raws]


def safe_load_configs() -> tuple[list[ModelConfig], list[DatasetConfig], list[str]]:
    """UI 启动时调用：失败也不抛异常，返回 warnings 供界面展示."""
    warnings: list[str] = []
    try:
        models = load_models_config()
    except Exception as e:
        warnings.append(f"模型配置加载失败: {e}")
        models = []
    try:
        datasets = load_datasets_config()
    except Exception as e:
        warnings.append(f"数据集配置加载失败: {e}")
        datasets = []
    return models, datasets, warnings


def find_by_name(items: list, name: str) -> Optional[Any]:
    for it in items:
        if getattr(it, "display_name", None) == name:
            return it
    return None


SOURCE_LABEL = {"local": "本地路径", "huggingface": "HuggingFace",
                "modelscope": "魔搭 ModelScope", "auto": "自动判断"}


def model_source(cfg: ModelConfig) -> str:
    """判定模型来源：显式配置优先，auto 则看本地是否存在该路径."""
    s = (cfg.source or "auto").lower()
    if s in ("local", "huggingface", "modelscope"):
        return s
    mid = cfg.model_id
    if mid.startswith((".", "/", "~")) or Path(mid).is_absolute():
        return "local"
    if (PROJECT_ROOT / mid).exists():
        return "local"
    return "huggingface"


def ensure_local_model(cfg: ModelConfig) -> str:
    """返回可直接喂给 FastLanguageModel.from_pretrained 的地址.

    - 本地：解析后的绝对路径（不存在则报错）
    - HuggingFace：ID 原样返回（首次使用自动下载）
    - 魔搭：snapshot_download 落到本地缓存后返回目录（需 pip install modelscope）
    """
    src = model_source(cfg)
    if src == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "要用魔搭模型请先安装: pip install modelscope") from e
        return snapshot_download(cfg.model_id.strip())
    if src == "local":
        p = cfg.resolved_model_id()
        if not Path(p).exists():
            raise FileNotFoundError(
                f"本地模型路径未找到: {cfg.model_id}（解析为 {p}）。"
                f"请在「模型管理」里改成正确的路径。")
        return p
    if not cfg.model_id or not cfg.model_id.strip():
        raise ValueError(f"模型 '{cfg.display_name}' 的 ID 为空。")
    return cfg.model_id.strip()


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
                       path: Path | str = PROJECT_ROOT / "models.json") -> None:
    """原子写回 models.json（先写临时文件再替换，防写一半崩）。"""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([model_to_dict(m) for m in models], f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
