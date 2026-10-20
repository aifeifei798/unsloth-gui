"""模型加载小工具（保持 stdlib-only，不导入 torch/transformers）。"""
from __future__ import annotations

import importlib.util
import os


def _socksio_available() -> bool:
    try:
        return importlib.util.find_spec("socksio") is not None
    except (ImportError, ValueError):
        return False


def sanitize_proxy_env() -> list[str]:
    """清掉 httpx 认不出的代理变量，返回被清掉的 "名字=值" 列表。

    常见环境是 `ALL_PROXY=socks://127.0.0.1:10808` + `http_proxy=http://...`。
    httpx 只认 http/https/socks5/socks5h，遇到 `socks://` 直接抛
    "Unknown scheme for proxy URL"，huggingface_hub 随即整个下载失败。
    这里把不认的 all_proxy 删掉，剩下的 http_proxy/https_proxy 照常生效
    （httpx 对具体协议的代理优先于 all:// 通配）。
    """
    removed: list[str] = []
    for name in ("ALL_PROXY", "all_proxy"):
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        scheme = raw.split("://", 1)[0].lower() if "://" in raw else ""
        if scheme in ("http", "https"):
            continue
        if scheme in ("socks5", "socks5h") and _socksio_available():
            continue
        os.environ.pop(name, None)
        removed.append(f"{name}={raw}")
    return removed


def configure_hf_transfer(enabled: bool) -> None:
    """在导入/调用 Unsloth 前设置 hf_transfer 开关。

    Unsloth 导入时只要没看到 HF_HUB_ENABLE_HF_TRANSFER 就会自动打开；
    这里显式写 "0" 可彻底关掉。默认关闭，只有模型配置显式开启时才用 "1"。
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if enabled else "0"


def configure_unsloth_env() -> None:
    """Unsloth 启动环境开关（默认关掉它的联网体检）。

    Unsloth 每次 from_pretrained 都会跑 get_statistics()：起子进程去 HuggingFace
    下载一个 README，看「HF 是不是挂了」（见 unsloth/models/_utils.py）。
    代理慢的时候这一步要 1~2 分钟（实测 76 秒），和训练本身毫无关系。
    UNSLOTH_DISABLE_STATISTICS=1 直接跳过；这里强制写死，不给环境变量开回去的余地
    （真要开，用 UNSLOTH_GUI_ALLOW_STATISTICS=1）。
    """
    if os.environ.get("UNSLOTH_GUI_ALLOW_STATISTICS", "0") == "1":
        return
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"


def _statistics_noop(*args, **kwargs) -> None:
    """Unsloth 联网体检的空实现。"""
    return None


def block_unsloth_statistics(modules=None) -> int:
    """把 Unsloth 的联网体检彻底堵死，返回被替换的入口数。

    光设 UNSLOTH_DISABLE_STATISTICS 不够：Unsloth 内部是
    `from ._utils import *`（导入时就把名字绑好了），所以要把所有带
    get_statistics 属性的 unsloth 模块都替换掉。必须在 import unsloth 之后调用。

    这只影响遥测和「HF 是否挂了」的探测，不影响加载 / 训练 / 推理。
    """
    if modules is None:
        import sys
        modules = list(sys.modules.values())
    patched = 0
    for mod in modules:
        if mod is None:
            continue
        name = getattr(mod, "__name__", "") or ""
        if not name.startswith("unsloth"):
            continue
        if getattr(mod, "get_statistics", None) is None:
            continue
        try:
            setattr(mod, "get_statistics", _statistics_noop)
            patched += 1
        except Exception:
            continue
    return patched


def unwrap_text_tokenizer(tokenizer):
    """多模态模型返回 Processor 时，取它内部的纯文本 Tokenizer。

    本项目只做文本 SFT/推理；Qwen3.5 等模型的 Processor 直接调用会把文本
    当成图片输入，且缺少 pad_token_id / eos_token_id。统一解包后，训练和
    推理都按普通文本 tokenizer 工作。
    """
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode"):
        return inner
    return tokenizer
