"""训练任务管理：修复原来“UI线程直接 trainer.train() 卡死、无法取消、断点逻辑粗糙”.

- 单进程单训练锁：防止并发训练互相踩显存
- CancelCallback：点“停止”后当前 epoch/step 结束即停，不杀进程
- TRL 新旧 API 兼容：SFTConfig+processing_class 优先，旧版自动回退
- resume：bool -> 自动找最新 checkpoint 路径
- 输出目录保护：同名且非续训直接报错，避免静默覆盖
"""
from __future__ import annotations

import gc
import inspect
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_PARENT_DIR = PROJECT_ROOT / "outputs"
LOGS_PARENT_DIR = PROJECT_ROOT / "logs"

_lock = threading.Lock()
_state = {"running": False, "experiment": None}
_cancel_event = threading.Event()


def is_training() -> bool:
    return bool(_state["running"])


def current_experiment() -> Optional[str]:
    return _state.get("experiment")


def request_cancel() -> str:
    if not _state["running"]:
        return "当前没有正在运行的训练。"
    _cancel_event.set()
    return f"已请求停止 '{_state.get('experiment')}'，将在当前 step 结束后停下并保存。"


def _find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.is_dir():
        return None
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def _build_training_args(training_mode: str, num_epochs: float, max_steps: int,
                         save_steps: int, output_dir: Path, logging_dir: Path,
                         batch_size: int, grad_accum: int, lr: float):
    """优先 SFTConfig（新TRL），否则回退 TrainingArguments（旧版）."""
    import torch

    try:
        from trl import SFTConfig as ArgsCls
        is_sft = True
    except Exception:
        from transformers import TrainingArguments as ArgsCls
        is_sft = False

    bf16_ok = False
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
        logging_steps=1,
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
    if training_mode == "按轮次 (Epochs)":
        base["num_train_epochs"] = float(num_epochs)
        base["save_strategy"] = "epoch"
    else:
        base["max_steps"] = int(max_steps)
        base["save_strategy"] = "steps"
        base["save_steps"] = int(save_steps)

    if is_sft:
        # 新版字段名 + 2026.7.x padding_free 显式关闭避免误触发
        base["max_length"] = int(os.environ.get("UNSLOTH_MAX_SEQ_LEN", "2048"))
        base["packing"] = False
        base["padding_free"] = False
        sig = inspect.signature(ArgsCls)
        base = {k: v for k, v in base.items() if k in sig.parameters}
    return ArgsCls(**base)


def _build_trainer(model, tokenizer, dataset, training_args, max_seq_length: int):
    from trl import SFTTrainer

    try:
        from transformers import TrainerCallback

        class CancelCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if _cancel_event.is_set():
                    control.should_training_stop = True
                return control

        cancel_cb = CancelCallback()
    except Exception:
        cancel_cb = None

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
    # dataset_text_field 在新版挪到了 SFTConfig，这里按需传递
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "dataset_num_proc" in params:
        kwargs["dataset_num_proc"] = 2
    if "packing" in params and "packing" not in kwargs:
        kwargs["packing"] = False
    try:
        from transformers.integrations import TensorBoardCallback
        kwargs.setdefault("callbacks", []).append(TensorBoardCallback())
    except Exception:
        pass
    if cancel_cb is not None:
        kwargs.setdefault("callbacks", []).append(cancel_cb)
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


def run_training(req: TrainRequest, progress=None) -> Generator[str, None, None]:
    """生成器：yield 状态文本，Gradio 侧保持响应 + 可取消."""
    exp = (req.experiment_name or "").strip().replace(" ", "_")
    if not exp:
        yield "错误：实验名称不能为空。"
        return
    if not req.selected_dataset_names:
        yield "错误：必须选择至少一个数据集。"
        return

    if not _lock.acquire(blocking=False):
        yield f"错误：已有训练 '{_state.get('experiment')}' 在运行，请先停止或等待完成。"
        return
    _state.update(running=True, experiment=exp)
    _cancel_event.clear()
    try:
        from .config import load_datasets_config, load_models_config, find_by_name
        from .config import ensure_local_model
        from .dataset_utils import prepare_dataset, combine_datasets, apply_chat_template_if_needed

        yield f"准备实验: {exp}"
        output_dir = OUTPUTS_PARENT_DIR / exp
        logging_dir = LOGS_PARENT_DIR / exp
        if output_dir.exists() and any(output_dir.iterdir()) and not req.resume_training:
            yield (
                f"错误：输出目录 '{output_dir}' 已存在且非空。为防覆盖请换实验名，"
                f"或勾选“从断点继续训练”。"
            )
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        models = load_models_config()
        datasets_cfg = load_datasets_config()
        model_cfg = find_by_name(models, req.selected_model_name)
        if model_cfg is None:
            yield f"错误：找不到模型 '{req.selected_model_name}'。"
            return

        if progress is not None:
            try:
                progress(0.1, desc="加载并处理数据集...")
            except Exception:
                pass
        yield "加载并处理数据集中..."
        all_ds = []
        for name in req.selected_dataset_names:
            cfg = find_by_name(datasets_cfg, name)
            if cfg is None:
                yield f"警告：找不到数据集配置 '{name}'，已跳过。"
                continue
            if not getattr(cfg, "processed", False):
                yield (f"错误：数据集 '{name}' 未经过「数据处理」制成统一数据，"
                       f"不能直接训练。请先去「数据处理」Tab 处理它。")
                return
            all_ds.append(prepare_dataset(cfg, req.truncate_dataset, max_samples=req.max_samples))
        if not all_ds:
            yield "错误：无法加载所选的数据集配置。"
            return
        combined = combine_datasets(all_ds)
        yield f"数据集就绪：共 {len(combined)} 条（{len(all_ds)} 个数据集合并）。"

        # 延迟导入重型依赖，UI 无 GPU 也能先打开；
        # torch 只在真正要碰模型时才导入，前面的数据校验不需要它
        import torch
        try:
            torch._dynamo.config.recompile_limit = 100
        except Exception:
            pass
        from unsloth import FastLanguageModel

        if progress is not None:
            try:
                progress(0.3, desc=f"初始化模型: {model_cfg.model_id}...")
            except Exception:
                pass
        yield f"初始化模型: {model_cfg.model_id} ..."
        try:
            model_path = ensure_local_model(model_cfg)
        except Exception as e:
            yield f"❌ 模型地址解析失败: {e}"
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
        tokenizer = apply_chat_template_if_needed(tokenizer, model_cfg) if hasattr(model_cfg, "chat_template") else tokenizer
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

        os.environ["UNSLOTH_MAX_SEQ_LEN"] = str(seq_len)
        training_args = _build_training_args(
            req.training_mode, req.num_epochs, req.max_steps, req.save_steps,
            output_dir, logging_dir, req.batch_size, req.grad_accum, req.lr,
        )
        trainer = _build_trainer(model, tokenizer, combined, training_args, seq_len)

        resume_arg = None
        if req.resume_training:
            latest = _find_latest_checkpoint(output_dir)
            resume_arg = latest or True
            yield f"从断点继续训练... ({resume_arg})"
        else:
            yield "开始新的训练...（点“停止训练”可在当前 step 后安全中断）"

        if progress is not None:
            try:
                progress(0.5, desc="训练中...")
            except Exception:
                pass
        trainer.train(resume_from_checkpoint=resume_arg)

        if _cancel_event.is_set():
            ckpt = _find_latest_checkpoint(output_dir)
            yield f"已手动停止。断点已保存{(f': {ckpt}') if ckpt else ''}，可勾选续训继续。"
            return

        if progress is not None:
            try:
                progress(0.9, desc="保存 LoRA 适配器...")
            except Exception:
                pass
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        yield f"训练完成！LoRA 已保存到 '{output_dir}'"
    except Exception:
        yield "❌ 训练失败:\n" + traceback.format_exc(limit=8)
    finally:
        # 释放显存，同一进程后续可做推理
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _state.update(running=False, experiment=None)
        _cancel_event.clear()
        _lock.release()
