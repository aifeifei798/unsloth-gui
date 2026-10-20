"""集中路径解析。

所有落盘位置都从这里取，避免每个模块各自 `Path(__file__).parent.parent`。
测试或便携部署可以用环境变量 `UNSLOTH_GUI_ROOT` 把整个根目录指到别处。
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    override = (os.environ.get("UNSLOTH_GUI_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _resolve_root()

MODELS_CONFIG = PROJECT_ROOT / "models.json"
DATASETS_CONFIG_DIR = PROJECT_ROOT / "datasets_config"
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"
PROCESSED_ROOT = LOCAL_DATA_DIR / "processed"
UPLOAD_ROOT = LOCAL_DATA_DIR / "uploads"
OUTPUTS_PARENT_DIR = PROJECT_ROOT / "outputs"
LOGS_PARENT_DIR = PROJECT_ROOT / "logs"
LOCALES_DIR = PROJECT_ROOT / "locales"
TRAIN_LOCK_FILE = LOCAL_DATA_DIR / ".gpu.lock"


def ensure_runtime_dirs() -> None:
    """把运行期目录建好（不会碰 datasets_config / models.json）。"""
    for p in (LOCAL_DATA_DIR, PROCESSED_ROOT, UPLOAD_ROOT, OUTPUTS_PARENT_DIR,
              LOGS_PARENT_DIR):
        p.mkdir(parents=True, exist_ok=True)


def resolve_local(value: str | Path) -> Path:
    """把用户输入的路径按「项目根目录」解析，支持 ~ 和绝对路径。"""
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def is_under(path: str | Path, root: str | Path = PROCESSED_ROOT) -> bool:
    """判断 path 是否在 root 下（防 ../ 逃逸）。"""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def safe_name(name: str) -> str:
    """只保留单层名字，拒绝路径分隔符与 ..。"""
    name = (name or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"invalid name: {name!r}")
    return name
