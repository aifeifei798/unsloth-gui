"""模型加载小工具（保持 stdlib-only，不导入 torch/transformers）。"""
from __future__ import annotations

import os


def configure_hf_transfer(enabled: bool) -> None:
    """在导入/调用 Unsloth 前设置 hf_transfer 开关。

    Unsloth 导入时只要没看到 HF_HUB_ENABLE_HF_TRANSFER 就会自动打开；
    这里显式写 "0" 可彻底关掉。默认关闭，只有模型配置显式开启时才用 "1"。
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if enabled else "0"


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
