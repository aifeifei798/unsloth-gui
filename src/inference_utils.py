"""推理管理：修复原来 State 存大对象、写死 cuda、缺 for_inference、历史拼接 bug."""
from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Generator, Optional

from .config import PROJECT_ROOT
from .i18n import t

OUTPUTS_PARENT_DIR = PROJECT_ROOT / "outputs"

_mgr_lock = threading.Lock()
_mgr = {"model": None, "tokenizer": None, "base": None, "lora": None}


def _looks_like_lora(d: Path) -> bool:
    """只有含 adapter 产物的目录才算 LoRA，失败残留的空目录不列出来."""
    try:
        if (d / "adapter_config.json").is_file():
            return True
        return any(f.suffix == ".safetensors" for f in d.iterdir() if f.is_file())
    except Exception:
        return False


def list_trained_loras() -> list[str]:
    if not OUTPUTS_PARENT_DIR.exists():
        return []
    return sorted([d.name for d in OUTPUTS_PARENT_DIR.iterdir()
                   if d.is_dir() and _looks_like_lora(d)])


def loaded_info() -> str:
    if _mgr["model"] is None:
        return t("inf.not_loaded")
    return t("inf.status_loaded", base=_mgr["base"], lora=_mgr["lora"])


def unload_model() -> str:
    with _mgr_lock:
        _mgr.update(model=None, tokenizer=None, base=None, lora=None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return t("inf.unloaded")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda"


def load_inference_model(base_model_name: str, lora_name: str, progress=None) -> str:
    if not base_model_name or not lora_name:
        return t("inf.need_both")
    from .config import load_models_config, find_by_name

    model_cfg = find_by_name(load_models_config(), base_model_name)
    if model_cfg is None:
        return t("inf.no_base", name=base_model_name)
    lora_path = OUTPUTS_PARENT_DIR / lora_name
    if not lora_path.exists():
        return t("inf.no_lora", path=lora_path)

    # 同模型同 LoRA 已加载则跳过
    if _mgr["model"] is not None and _mgr["base"] == base_model_name and _mgr["lora"] == lora_name:
        return t("inf.already", name=lora_name)

    with _mgr_lock:
        # 先卸旧模型防 OOM
        _mgr.update(model=None, tokenizer=None, base=None, lora=None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if progress is not None:
                try:
                    progress(0.2, desc=t("inf.loading_base", id=model_cfg.model_id))
                except Exception:
                    pass
            from unsloth import FastLanguageModel
            from .config import ensure_local_model

            try:
                model_path = ensure_local_model(model_cfg)
            except Exception as e:
                return t("inf.resolve_fail", err=e)
            seq_len = int(getattr(model_cfg, "max_seq_length", 2048) or 2048)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=seq_len,
                load_in_4bit=bool(model_cfg.load_in_4bit),
            )
            # 兼容 chat_template
            if getattr(model_cfg, "chat_template", None):
                try:
                    from unsloth.chat_templates import get_chat_template
                    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg.chat_template)
                except Exception:
                    pass
            if progress is not None:
                try:
                    progress(0.6, desc=t("inf.loading_lora", name=lora_name))
                except Exception:
                    pass
            loaded = False
            if hasattr(model, "load_adapter"):
                try:
                    model.load_adapter(str(lora_path))
                    loaded = True
                except Exception:
                    loaded = False
            if not loaded:
                # 回退：PEFT 方式挂载
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, str(lora_path))
                    loaded = True
                except Exception as e:
                    return t("inf.mount_fail", err=e)
            try:
                FastLanguageModel.for_inference(model)
            except Exception:
                pass
            _mgr.update(model=model, tokenizer=tokenizer, base=base_model_name, lora=lora_name)
            return t("inf.loaded", name=lora_name)
        except Exception as e:
            import traceback
            return t("inf.load_fail", err=e, trace=traceback.format_exc(limit=6))


def run_chat(user_input: str, history: Optional[list], system_prompt: str,
             max_new_tokens: int = 256, temperature: float = 0.7,
             top_p: float = 0.95, top_k: int = 40) -> Generator[list, None, None]:
    """流式生成器. history 为 OpenAI 式 [{"role","content"}]."""
    from threading import Thread

    history = list(history or [])
    if not (user_input or "").strip():
        yield history
        return
    model, tokenizer = _mgr["model"], _mgr["tokenizer"]
    if model is None or tokenizer is None:
        history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": t("inf.model_missing")},
        ]
        yield history
        return

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    history = history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": ""}]

    try:
        from transformers import TextIteratorStreamer
    except Exception:
        try:
            from transformers.generation.streamers import TextIteratorStreamer
        except Exception as e:
            history[-1]["content"] = t("inf.no_streamer", err=e)
            yield history
            return

    try:
        prompt_inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # 模型无 chat_template 时回退为简单拼接
        flat = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        prompt_inputs = tokenizer(flat, return_tensors="pt")["input_ids"]
    try:
        prompt_inputs = prompt_inputs.to(_device())
    except Exception:
        pass

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=prompt_inputs, streamer=streamer,
        max_new_tokens=int(max_new_tokens), do_sample=True,
        temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
        repetition_penalty=1.1, use_cache=True,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()
    yield history
    for new_text in streamer:
        history[-1]["content"] += new_text
        yield history
