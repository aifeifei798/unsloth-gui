"""推理管理：按训练时的 run_manifest.json 还原 prompt 格式，流式对话，可取消.

修复点：
- 不再用 tokenizer.chat_template 推理（训练数据是纯文本模板，两边格式必须一致）
- 历史按 token 截断，长对话不会顶爆上下文
- 显式传 attention_mask，消除 decoder-only 生成警告
- 生成锁：同一时刻只允许一个生成，避免并发抢显存
- 可取消：WebSocket 断开或点停止都会让 generate 在当前 step 后结束
- 设备自适应，没 CUDA 也能加载（CPU 慢但可用）
"""
from __future__ import annotations

import gc
import json
import queue
import threading
from pathlib import Path
from typing import Generator, Optional

from .i18n import t
from .model_utils import configure_hf_transfer, unwrap_text_tokenizer
from .paths import OUTPUTS_PARENT_DIR

_mgr_lock = threading.Lock()
_gen_lock = threading.Lock()
_mgr: dict = {
    "model": None, "tokenizer": None, "base": None, "lora": None,
    "base_model_id": None, "seq_len": 2048, "prompt_template": "",
    "prompt_mode": "template",
    "cancel_event": None,
}

DEFAULT_INSTRUCTION_HEADER = "### Instruction:"
DEFAULT_RESPONSE_HEADER = "### Response:"
DEFAULT_INPUT_HEADER = "### Input:"


def _looks_like_lora(d: Path) -> bool:
    """只有含 adapter 产物的目录才算 LoRA，失败残留的空目录不列出来."""
    try:
        if (d / "adapter_config.json").is_file():
            return True
        return ((d / "adapter_model.safetensors").is_file()
                or (d / "adapter_model.bin").is_file())
    except Exception:
        return False


def list_trained_loras() -> list[str]:
    if not OUTPUTS_PARENT_DIR.exists():
        return []
    return sorted([d.name for d in OUTPUTS_PARENT_DIR.iterdir()
                   if d.is_dir() and _looks_like_lora(d)])


def read_run_manifest(lora_name: str) -> dict:
    path = OUTPUTS_PARENT_DIR / lora_name / "run_manifest.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def lora_info(lora_name: str) -> dict:
    """给 UI 的只读摘要：基础模型、seq_len、数据集、训练时间."""
    man = read_run_manifest(lora_name)
    out = {
        "name": lora_name,
        "base_model": man.get("base_model", ""),
        "base_model_display": man.get("base_model_display", ""),
        "seq_len": man.get("seq_len", ""),
        "datasets": man.get("datasets", []),
        "status": man.get("status", ""),
        "prompt_mode": man.get("prompt_mode", "template"),
        "has_manifest": bool(man),
    }
    return out


def loaded_info() -> str:
    if _mgr["model"] is None:
        return t("inf.not_loaded")
    return t("inf.status_loaded", base=_mgr["base"], lora=_mgr["lora"])


def loaded_payload() -> dict:
    return {
        "loaded": _mgr["model"] is not None,
        "base": _mgr["base"],
        "lora": _mgr["lora"],
        "seq_len": _mgr["seq_len"],
        "has_template": bool(_mgr.get("prompt_template")),
        "prompt_mode": _mgr.get("prompt_mode", "template"),
    }


def unload_model() -> str:
    ev = _mgr.get("cancel_event")
    if ev is not None:
        ev.set()
    with _mgr_lock:
        _mgr.update(model=None, tokenizer=None, base=None, lora=None,
                    base_model_id=None, seq_len=2048, prompt_template="",
                    prompt_mode="template",
                    cancel_event=None)
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
        return "cpu"


def _headers_from_template(template: str) -> tuple[str, str, bool]:
    """从训练模板里取段落标题，兼容自定义标题；标准模板走默认值."""
    if not template:
        return DEFAULT_INSTRUCTION_HEADER, DEFAULT_RESPONSE_HEADER, False
    has_input = "{input}" in template
    ins = DEFAULT_INSTRUCTION_HEADER
    resp = DEFAULT_RESPONSE_HEADER
    try:
        before_ins = template.split("{instruction}")[0].strip().splitlines()
        if before_ins:
            ins = before_ins[-1].strip() or ins
        before_out = template.split("{output}")[0]
        marker = before_out.rsplit("\n", 1)
        if len(marker) == 2:
            resp = marker[0].splitlines()[-1].strip() or resp
    except Exception:
        pass
    return ins, resp, has_input


def _render_prompt(history: list, user_input: str, system_prompt: str,
                   ins_header: str, resp_header: str, has_input: bool) -> str:
    """把多轮对话还原成训练时的纯文本模板格式.

    - system_prompt 只在第一轮进 `### Input:`（模板没有 Input 段就并进 Instruction）
    - 历史按 user/assistant 交替渲染
    - 末尾补 `### Response:` 让模型接着写
    """
    turns: list[dict] = []
    for m in history or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})
    turns.append({"role": "user", "content": (user_input or "").strip()})

    system = (system_prompt or "").strip()
    blocks: list[str] = []
    first_user = True
    for m in turns:
        if m["role"] == "user":
            instruction = m["content"]
            if first_user and system and not has_input:
                instruction = f"{system}\n\n{instruction}"
            parts = [f"{ins_header}\n{instruction}"]
            if first_user and system and has_input:
                parts.append(f"{DEFAULT_INPUT_HEADER}\n{system}")
            blocks.append("\n\n".join(parts))
            first_user = False
        else:
            blocks.append(f"{resp_header}\n{m['content']}")
    # 最后必须是空的 Response 段
    if turns and turns[-1]["role"] == "assistant":
        blocks.append(f"{resp_header}\n")
    else:
        blocks.append(f"{resp_header}")
    return "\n\n".join(blocks)


def _token_len(tokenizer, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return max(1, len(text) // 3)


def _build_prompt(tokenizer, history: list, user_input: str, system_prompt: str,
                  ins_header: str, resp_header: str, has_input: bool,
                  max_seq_len: int, max_new_tokens: int) -> tuple[str, int]:
    """按 token 预算从旧到新丢历史；返回 (prompt, 实际 max_new_tokens)."""
    budget = max(64, int(max_seq_len) - int(max_new_tokens) - 8)
    turns = list(history or [])
    while True:
        prompt = _render_prompt(turns, user_input, system_prompt,
                                ins_header, resp_header, has_input)
        n = _token_len(tokenizer, prompt)
        if n <= budget or not turns:
            break
        # 丢掉最早的一轮（user+assistant）
        drop = 0
        while drop < len(turns) and drop < 2:
            drop += 1
        turns = turns[drop:]
    out_tokens = max(1, min(int(max_new_tokens), int(max_seq_len) - n - 2))
    return prompt, out_tokens


# ------------------------------------------------------------ chat 模板模式

def _chat_messages(history: list, user_input: str, system_prompt: str) -> list[dict]:
    """历史 + 本轮输入 → chat messages（与训练时同一套角色）。"""
    messages: list[dict] = []
    system = (system_prompt or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    for m in history or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (user_input or "").strip()})
    return messages


def _render_chat_prompt(tokenizer, history: list, user_input: str,
                        system_prompt: str) -> str:
    """用模型自带 chat 模板渲染「轮到模型说话」的提示词。

    enable_thinking=True 时提示词停在 `<think>` 开头，和训练文本里
    assistant 段（<think>…</think>）的写法对齐。
    """
    from .dataset_utils import render_chat_text

    messages = _chat_messages(history, user_input, system_prompt)
    return render_chat_text(tokenizer, messages, add_generation_prompt=True,
                            enable_thinking=True)


def _chat_stop_string(tokenizer) -> str:
    """从模板里反推 assistant 起始标记，用作停止串（模型自己开新一轮就停）。"""
    try:
        probe = [{"role": "user", "content": "x"}]
        base = tokenizer.apply_chat_template(probe, tokenize=False)
        gen = tokenizer.apply_chat_template(probe, tokenize=False,
                                            add_generation_prompt=True)
        diff = str(gen)[len(str(base)):]
        first = diff.splitlines()[0].strip() if diff else ""
        return first if 0 < len(first) <= 64 else ""
    except Exception:
        return ""


def _build_chat_prompt(tokenizer, history: list, user_input: str,
                       system_prompt: str, max_seq_len: int,
                       max_new_tokens: int) -> tuple[str, int]:
    """chat 模板模式：按 token 预算丢历史，返回 (prompt, 实际 max_new_tokens)."""
    budget = max(64, int(max_seq_len) - int(max_new_tokens) - 8)
    turns = list(history or [])
    while True:
        prompt = _render_chat_prompt(tokenizer, turns, user_input, system_prompt)
        n = _token_len(tokenizer, prompt)
        if n <= budget or not turns:
            break
        drop = 0
        while drop < len(turns) and drop < 2:
            drop += 1
        turns = turns[drop:]
    out_tokens = max(1, min(int(max_new_tokens), int(max_seq_len) - n - 2))
    return prompt, out_tokens


class _CancelCriteria:
    """transformers StoppingCriteria 的鸭子类型版本，避免顶层 import torch."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


def stream_chat(user_input: str, history: Optional[list], system_prompt: str,
                max_new_tokens: int = 256, temperature: float = 0.7,
                top_p: float = 0.95, top_k: int = 40,
                cancel_event: Optional[threading.Event] = None,
                ) -> Generator[str, None, None]:
    """流式产出文本增量。模型未加载时抛 RuntimeError，由调用方转成界面提示."""
    text = (user_input or "").strip()
    if not text:
        return
    model, tokenizer = _mgr["model"], _mgr["tokenizer"]
    if model is None or tokenizer is None:
        raise RuntimeError(t("inf.model_missing"))
    if not _gen_lock.acquire(blocking=False):
        raise RuntimeError(t("inf.busy"))

    cancel = cancel_event or threading.Event()
    _mgr["cancel_event"] = cancel
    thread = None
    try:
        from threading import Thread

        from transformers import TextIteratorStreamer

        ins_header, resp_header, has_input = _headers_from_template(
            _mgr.get("prompt_template") or "")
        max_seq_len = int(_mgr.get("seq_len") or 2048)
        chat_mode = (_mgr.get("prompt_mode") or "template") == "chat_template"
        if chat_mode:
            prompt, out_tokens = _build_chat_prompt(
                tokenizer, list(history or []), text, system_prompt,
                max_seq_len, max_new_tokens)
        else:
            prompt, out_tokens = _build_prompt(
                tokenizer, list(history or []), text, system_prompt,
                ins_header, resp_header, has_input, max_seq_len, max_new_tokens)

        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = getattr(model, "device", None) or _device()
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
        prompt_len = int(inputs["input_ids"].shape[-1])
        if prompt_len + out_tokens > max_seq_len:
            keep = max(1, max_seq_len - out_tokens)
            inputs["input_ids"] = inputs["input_ids"][:, -keep:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -keep:]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600)
        gen_kwargs: dict = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            streamer=streamer,
            max_new_tokens=int(out_tokens),
            repetition_penalty=1.1,
            use_cache=True,
            stopping_criteria=[_CancelCriteria(cancel)],
            tokenizer=tokenizer,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )
        if float(temperature) > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature),
                              top_p=float(top_p))
            if int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)
        else:
            gen_kwargs["do_sample"] = False
        # 只把“下一轮 Instruction”当停止符。把 Response 也放进 stop_strings，
        # 某些模型第一 token 就复读标题，会立刻截断成空回复。
        # chat 模板模式改成「模型自己开新一轮 assistant」时停。
        if chat_mode:
            stop = _chat_stop_string(tokenizer)
            if stop:
                gen_kwargs["stop_strings"] = [stop]
        else:
            gen_kwargs["stop_strings"] = [ins_header]

        errors: list[BaseException] = []

        def _generate():
            import torch

            with torch.inference_mode():
                try:
                    model.generate(**gen_kwargs)
                except TypeError:
                    # 老版本 transformers 不认 stop_strings / tokenizer
                    gen_kwargs.pop("stop_strings", None)
                    gen_kwargs.pop("tokenizer", None)
                    try:
                        model.generate(**gen_kwargs)
                    except Exception as e:
                        errors.append(e)
                        streamer.end()
                except Exception as e:
                    errors.append(e)
                    streamer.end()

        thread = Thread(target=_generate, daemon=True)
        thread.start()
        try:
            for piece in streamer:
                if cancel.is_set():
                    break
                if piece:
                    yield piece
        except queue.Empty:
            pass
        if errors and not cancel.is_set():
            raise RuntimeError(str(errors[0])) from errors[0]
    finally:
        # 消费端提前退出/断线时，让生成线程尽快停；正常跑完则不要误标取消
        if thread is not None and thread.is_alive():
            cancel.set()
            thread.join(timeout=10)
        elif thread is not None:
            thread.join(timeout=0.1)
        _mgr["cancel_event"] = None
        _gen_lock.release()


def load_inference_model(base_model_name: str, lora_name: str, progress=None) -> str:
    if not base_model_name or not lora_name:
        return t("inf.need_both")
    from .config import ensure_local_model, find_by_name, load_models_config
    from .train_utils import is_training

    if is_training():
        return t("inf.training_busy")

    model_cfg = find_by_name(load_models_config(), base_model_name)
    if model_cfg is None:
        return t("inf.no_base", name=base_model_name)
    lora_path = OUTPUTS_PARENT_DIR / lora_name
    if not lora_path.exists():
        return t("inf.no_lora", path=lora_path)

    # 同模型同 LoRA 已加载则跳过
    if (_mgr["model"] is not None and _mgr["base"] == base_model_name
            and _mgr["lora"] == lora_name):
        return t("inf.already", name=lora_name)

    manifest = read_run_manifest(lora_name)
    seq_len = int(manifest.get("seq_len") or getattr(model_cfg, "max_seq_length", 2048)
                  or 2048)
    template = manifest.get("prompt_template") or ""
    # 老 manifest 没有 prompt_mode → 仍按 prompt_template 还原；新训练走 chat 模板
    prompt_mode = "chat_template" if manifest.get("prompt_mode") == "chat_template" \
        else "template"
    # 模板名以 manifest 为准（老 manifest 没有这个字段才回退模型配置）
    if "chat_template" in manifest:
        chat_template_name = manifest.get("chat_template") or ""
    else:
        chat_template_name = getattr(model_cfg, "chat_template", None) or ""
    warning = ""
    if manifest.get("base_model_display") and manifest["base_model_display"] != base_model_name:
        warning = "\n" + t("inf.base_mismatch",
                           want=manifest["base_model_display"], got=base_model_name)
    elif manifest.get("base_model") and model_cfg.model_id != manifest["base_model"]:
        warning = "\n" + t("inf.base_mismatch",
                           want=manifest["base_model"], got=model_cfg.model_id)
    if not template and manifest.get("prompt_templates") and prompt_mode == "template":
        warning += "\n" + t("inf.template_ambiguous")

    with _mgr_lock:
        # 先卸旧模型防 OOM
        _mgr.update(model=None, tokenizer=None, base=None, lora=None,
                    base_model_id=None, seq_len=2048, prompt_template="",
                    prompt_mode="template",
                    cancel_event=None)
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
            configure_hf_transfer(bool(model_cfg.fast_download))
            import unsloth  # 必须早于 transformers，否则 Unsloth 的优化不生效
            from unsloth import FastLanguageModel

            try:
                model_path = ensure_local_model(model_cfg)
            except Exception as e:
                return t("inf.resolve_fail", err=e)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=seq_len,
                load_in_4bit=bool(model_cfg.load_in_4bit),
            )
            tokenizer = unwrap_text_tokenizer(tokenizer)
            if chat_template_name:
                try:
                    from unsloth.chat_templates import get_chat_template
                    tokenizer = get_chat_template(
                        tokenizer, chat_template=chat_template_name)
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
            _mgr.update(model=model, tokenizer=tokenizer, base=base_model_name,
                        lora=lora_name, base_model_id=model_cfg.model_id,
                        seq_len=seq_len, prompt_template=template,
                        prompt_mode=prompt_mode,
                        cancel_event=None)
            return t("inf.loaded", name=lora_name) + warning
        except Exception as e:
            import traceback
            return t("inf.load_fail", err=e, trace=traceback.format_exc(limit=6))
