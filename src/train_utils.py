"""训练任务管理：单训练锁、可取消、断点续训、训练清单.

关键行为：
- 单进程单训练锁，防止并发训练互相踩显存
- CancelCallback：点“停止”后当前 step 结束即停，不杀进程
- 取消后仍会把当前 LoRA adapter 落盘，避免白训
- 续训按 checkpoint 数字取最新；勾了续训却没有断点直接报错，不静默重训覆盖
- 训练完成写 run_manifest.json（基础模型 / 数据集 / 模板 / seq_len），
  推理端据此还原训练时的 prompt 格式
- 无 CUDA 时给明确提示，不做无意义的模型加载
- 重型依赖全部函数内延迟导入
"""
from __future__ import annotations

import gc
import inspect
import re
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from .i18n import t
from .model_utils import configure_hf_transfer, sanitize_proxy_env, unwrap_text_tokenizer
from .paths import LOGS_PARENT_DIR, OUTPUTS_PARENT_DIR
from .storage import atomic_write_json

_lock = threading.Lock()
_state = {"running": False, "experiment": None}
_cancel_event = threading.Event()
_external_cancel: Optional[threading.Event] = None


def is_training() -> bool:
    return bool(_state["running"])


def current_experiment() -> Optional[str]:
    return _state.get("experiment")


def request_cancel() -> str:
    if not _state["running"]:
        return t("tr.cancel_none")
    _cancel_event.set()
    return t("tr.cancel_req", exp=_state.get("experiment"))


def is_epoch_mode(mode: str) -> bool:
    """兼容旧中文标签、英文标签与代码值（epochs / steps）."""
    s = str(mode or "").strip().lower()
    return s in ("epoch", "epochs") or "epoch" in s or "轮次" in s


def _find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """按 checkpoint-<数字> 的最大值取，不靠 mtime（复制/恢复后 mtime 不可靠）."""
    if not output_dir.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m:
            found.append((int(m.group(1)), p))
    if found:
        return str(max(found, key=lambda x: x[0])[1])
    # 极端兼容：非标准命名的 checkpoint 目录按 mtime 兜底
    ckpts = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if ckpts:
        return str(max(ckpts, key=lambda p: p.stat().st_mtime))
    return None


def _call_progress(progress, frac: float, desc: str) -> None:
    if progress is None:
        return
    try:
        progress(frac, desc=desc)
    except Exception:
        pass


def _build_training_args(training_mode: str, num_epochs: float, max_steps: int,
                         save_steps: int, output_dir: Path, logging_dir: Path,
                         batch_size: int, grad_accum: int, lr: float,
                         seq_len: int):
    """优先 SFTConfig（新 TRL），否则回退 TrainingArguments（旧版）."""
    import torch

    try:
        from trl import SFTConfig as ArgsCls
        is_sft = True
    except Exception:
        from transformers import TrainingArguments as ArgsCls
        is_sft = False

    try:
        bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        bf16_ok = False

    base = dict(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        learning_rate=float(lr),
        logging_steps=5,
        optim="adamw_8bit",
        fp16=not bf16_ok,
        bf16=bf16_ok,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_total_limit=3,
        report_to=["tensorboard"],
    )
    if is_epoch_mode(training_mode):
        base["num_train_epochs"] = float(num_epochs)
        base["save_strategy"] = "epoch"
    else:
        base["max_steps"] = int(max_steps)
        base["save_strategy"] = "steps"
        base["save_steps"] = int(save_steps)

    if is_sft:
        # 新版字段名 + 显式关闭 padding_free，避免误触发
        base["max_length"] = int(seq_len)
        base["packing"] = False
        base["padding_free"] = False
        base["dataset_text_field"] = "text"
        sig = inspect.signature(ArgsCls)
        base = {k: v for k, v in base.items() if k in sig.parameters}
    return ArgsCls(**base)


def _build_trainer(model, tokenizer, dataset, training_args, max_seq_length: int,
                   on_log=None):
    from trl import SFTTrainer

    callbacks = []
    try:
        from transformers import TrainerCallback

        class CancelCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if _cancel_event.is_set() or (
                        _external_cancel is not None and _external_cancel.is_set()):
                    control.should_training_stop = True
                return control

        callbacks.append(CancelCallback())

        if on_log is not None:
            class MetricsCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    payload = {"step": int(getattr(state, "global_step", 0) or 0),
                               "epoch": getattr(state, "epoch", None)}
                    payload.update({k: v for k, v in logs.items()
                                    if isinstance(v, (int, float, str, bool)) or v is None})
                    try:
                        on_log(payload)
                    except Exception:
                        pass

            callbacks.append(MetricsCallback())
    except Exception:
        pass

    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters
    kwargs: dict = dict(model=model, train_dataset=dataset, args=training_args)
    # tokenizer vs processing_class（新版 TRL 已改名）
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    # max_seq_length vs max_length
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = int(max_seq_length)
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "dataset_num_proc" in params:
        kwargs["dataset_num_proc"] = 2
    if "packing" in params:
        kwargs["packing"] = False
    if callbacks:
        kwargs["callbacks"] = callbacks
    return SFTTrainer(**kwargs)


@dataclass
class TrainRequest:
    experiment_name: str
    resume_training: bool
    truncate_dataset: bool
    max_samples: int
    training_mode: str
    num_epochs: float
    max_steps: int
    save_steps: int
    selected_model_name: str
    selected_dataset_names: list
    lora_r: int
    lora_alpha: int
    batch_size: int
    grad_accum: int
    lr: float
    max_seq_length: int
    use_chat_template: bool = True  # True=用模型自带 chat 模板，False=老 prompt_template


def _safe_exp_name(name: str) -> str:
    exp = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", (name or "").strip())
    exp = exp.strip("._")
    return exp[:80]


def _manifest_payload(req: TrainRequest, model_cfg, datasets_cfg, seq_len: int,
                      status: str, prompt_mode: str = "template") -> dict:
    templates: list[str] = []
    columns: list[dict] = []
    for cfg in datasets_cfg:
        if cfg.prompt_template and cfg.prompt_template not in templates:
            templates.append(cfg.prompt_template)
        columns.append({"name": cfg.display_name,
                        "input_columns": cfg.input_columns,
                        "template": cfg.prompt_template})
    return {
        "status": status,
        "base_model": model_cfg.model_id,
        "base_model_display": model_cfg.display_name,
        "base_model_source": model_cfg.source,
        "datasets": [c.display_name for c in datasets_cfg],
        "dataset_columns": columns,
        "prompt_mode": prompt_mode,
        # chat 模式下最终文本由模型自带 chat 模板渲染，不写死 Instruction/Response 模板
        "prompt_template": (templates[0] if len(templates) == 1 else "")
        if prompt_mode == "template" else "",
        "prompt_templates": templates,
        "chat_template": (getattr(model_cfg, "chat_template", None) or ""),
        "seq_len": int(seq_len),
        "training_mode": "epochs" if is_epoch_mode(req.training_mode) else "steps",
        "num_epochs": float(req.num_epochs),
        "max_steps": int(req.max_steps),
        "save_steps": int(req.save_steps),
        "lora_r": int(req.lora_r),
        "lora_alpha": int(req.lora_alpha),
        "batch_size": int(req.batch_size),
        "grad_accum": int(req.grad_accum),
        "learning_rate": float(req.lr),
        "truncated": bool(req.truncate_dataset),
    }


def _write_manifest(output_dir: Path, payload: dict) -> None:
    try:
        atomic_write_json(Path(output_dir) / "run_manifest.json", payload)
    except Exception:
        pass


def run_training(req: TrainRequest, progress=None,
                 on_log=None, cancel_event=None) -> Generator[str, None, None]:
    """生成器：yield 状态文本；progress(frac, desc=) 推进度；on_log(dict) 推指标."""
    global _external_cancel
    exp = _safe_exp_name(req.experiment_name)
    if not exp:
        yield t("tr.no_exp_name")
        return
    if not req.selected_dataset_names:
        yield t("tr.no_datasets")
        return

    if not _lock.acquire(blocking=False):
        yield t("tr.running", exp=_state.get("experiment"))
        return
    _state.update(running=True, experiment=exp)
    _cancel_event.clear()
    _external_cancel = cancel_event
    output_dir = OUTPUTS_PARENT_DIR / exp
    logging_dir = LOGS_PARENT_DIR / exp
    # 数据集/模型都要走网络，先把 httpx 认不出的代理变量清掉
    sanitize_proxy_env()

    def _cancelled() -> bool:
        return (_cancel_event.is_set()
                or (_external_cancel is not None and _external_cancel.is_set()))

    try:
        from .config import (
            ensure_local_model,
            find_by_name,
            is_trainable_dataset,
            load_datasets_config,
            load_models_config,
            trainable_reason,
        )
        from .dataset_utils import (
            combine_datasets,
            format_with_chat_template,
            format_with_template,
            has_chat_template,
            load_raw_dataset,
            supports_chat_roles,
        )

        yield t("tr.prep_exp", exp=exp)

        if _cancelled():
            yield t("tr.cancelled_before_start")
            return

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        non_empty = output_dir.exists() and any(output_dir.iterdir())
        resume_arg = None
        if non_empty:
            if not req.resume_training:
                yield t("tr.dir_exists", dir=output_dir)
                return
            resume_arg = _find_latest_checkpoint(output_dir)
            if resume_arg is None:
                yield t("tr.resume_no_ckpt", dir=output_dir)
                return

        models = load_models_config()
        dataset_warnings: list[str] = []
        datasets_cfg = load_datasets_config(warnings=dataset_warnings)
        for w in dataset_warnings:
            yield w
        model_cfg = find_by_name(models, req.selected_model_name)
        if model_cfg is None:
            yield t("tr.no_model", name=req.selected_model_name)
            return

        _call_progress(progress, 0.1, t("tr.loading_data"))
        yield t("tr.loading_data")
        raw_ds = []
        selected_cfgs = []
        for name in req.selected_dataset_names:
            if _cancelled():
                yield t("tr.cancelled_before_start")
                return
            cfg = find_by_name(datasets_cfg, name)
            if cfg is None:
                yield t("tr.warn_skip", name=name)
                continue
            if not is_trainable_dataset(cfg):
                reason = trainable_reason(cfg) or t("tr.unprocessed", name=name)
                yield t("tr.unprocessed_reason", name=name, reason=reason)
                return
            # 先只加载原始数据：真正的格式化要等 tokenizer 到位（chat 模板模式）
            raw_ds.append(load_raw_dataset(cfg, req.truncate_dataset,
                                           max_samples=req.max_samples))
            selected_cfgs.append(cfg)
        if not raw_ds:
            yield t("tr.no_valid_ds")
            return
        yield t("tr.ready", n=sum(len(d) for d in raw_ds), k=len(raw_ds))

        # 重型依赖最后才导入：前面的数据校验完全不需要 torch
        import torch

        if not torch.cuda.is_available():
            yield t("tr.no_cuda")
            return
        try:
            torch._dynamo.config.recompile_limit = 100
        except Exception:
            pass
        configure_hf_transfer(bool(model_cfg.fast_download))
        sanitize_proxy_env()  # socks:// 代理会让 huggingface_hub 直接崩
        import unsloth  # 必须早于 transformers，否则 Unsloth 的优化不生效
        from unsloth import FastLanguageModel

        _call_progress(progress, 0.3, t("tr.init_model", id=model_cfg.model_id))
        yield t("tr.init_model", id=model_cfg.model_id)
        try:
            model_path = ensure_local_model(model_cfg)
        except Exception as e:
            yield t("tr.resolve_fail", err=e)
            return
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        dtype = dtype_map.get(model_cfg.dtype) if model_cfg.dtype else None
        seq_len = int(req.max_seq_length or model_cfg.max_seq_length or 2048)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=seq_len,
            dtype=dtype,
            load_in_4bit=bool(model_cfg.load_in_4bit),
        )
        tokenizer = unwrap_text_tokenizer(tokenizer)
        # 模型配置指定了 Unsloth 模板名（chatml / qwen-2.5 / llama-3.1 …）就用它，
        # 否则用模型自带的 chat_template（与 Unsloth 官方用法一致）
        if model_cfg.chat_template:
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(
                    tokenizer, chat_template=model_cfg.chat_template)
            except Exception:
                pass
        if _cancelled():
            yield t("tr.cancelled_before_start")
            return

        # 决定最终训练文本格式：模型自带 chat 模板（新，默认）还是老 prompt_template
        use_chat = bool(req.use_chat_template)
        if use_chat and not has_chat_template(tokenizer):
            yield t("tr.no_chat_template")
            use_chat = False
        if use_chat:
            bad = [c.display_name for c in selected_cfgs if not supports_chat_roles(c)]
            if bad:
                yield t("tr.chat_fallback", names=", ".join(bad))
                use_chat = False
        prompt_mode = "chat_template" if use_chat else "template"
        yield t("tr.prompt_mode", mode=prompt_mode)
        formatted = []
        for ds, cfg in zip(raw_ds, selected_cfgs):
            if use_chat:
                formatted.append(format_with_chat_template(ds, cfg, tokenizer))
            else:
                formatted.append(format_with_template(ds, cfg))
        combined = combine_datasets(formatted)

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(req.lora_r),
            lora_alpha=int(req.lora_alpha),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        training_args = _build_training_args(
            req.training_mode, req.num_epochs, req.max_steps, req.save_steps,
            output_dir, logging_dir, req.batch_size, req.grad_accum, req.lr,
            seq_len,
        )
        trainer = _build_trainer(model, tokenizer, combined, training_args,
                                 seq_len, on_log=on_log)

        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        if resume_arg:
            yield t("tr.resume_from", ckpt=resume_arg)
        else:
            yield t("tr.resume_fresh") if req.resume_training else t("tr.start_new")

        _call_progress(progress, 0.5, t("tr.training"))
        trainer.train(resume_from_checkpoint=resume_arg)

        stopped = _cancelled()
        if stopped:
            _call_progress(progress, 0.85, t("tr.saving"))
            yield t("tr.saving")
        else:
            _call_progress(progress, 0.9, t("tr.saving"))
            yield t("tr.saving")

        # 无论正常结束还是手动停止，都把当前 adapter 落盘
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        if stopped:
            try:
                trainer.save_state()
            except Exception:
                pass
        status = "stopped" if stopped else "completed"
        _write_manifest(output_dir, _manifest_payload(
            req, model_cfg, selected_cfgs, seq_len, status, prompt_mode=prompt_mode))

        if stopped:
            ckpt = _find_latest_checkpoint(output_dir)
            yield t("tr.stopped",
                    ckpt=t("tr.stopped_ckpt", ckpt=ckpt) if ckpt else "")
            return
        _call_progress(progress, 1.0, t("tr.done", dir=output_dir))
        yield t("tr.done", dir=output_dir)
    except Exception:
        yield t("tr.failed", trace=traceback.format_exc(limit=8))
    finally:
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _state.update(running=False, experiment=None)
        _cancel_event.clear()
        _external_cancel = None
        _lock.release()
