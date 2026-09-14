"""
Unsloth GUI Trainer & Playground (v4.0 - 模块化重构版)

修复 v3.8 的主要问题：
- 单文件巨石 -> src/ 模块化（config/dataset/train/inference/tensorboard）
- 训练阻塞 UI -> 生成器 + 取消回调 + 单训练锁
- 旧 TRL API -> SFTConfig/processing_class 优先、旧版自动回退
- 数据集脆弱 -> 列校验、schema 对齐、安全 format、预览
- 推理写死 cuda/历史 bug -> 设备自适应、for_inference、正确多轮
- TensorBoard 写死端口/sleep -> 端口探测、复用、退出清理
- share=True 默认公网暴露 -> 默认关闭，可用参数开启
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr

from src.config import PROJECT_ROOT, find_by_name, safe_load_configs
from src.config import load_models_config, save_models_config, ModelConfig, model_source
from src.config import source_label
from src.i18n import get_lang, set_lang, t
from src.data_mgmt import delete_entry, detail_text, list_entries, rename_entry, sample_text
from src.dataprep import generate_unified, inspect_text, persist_upload, preview_row
from src.dataset_utils import dataset_preview_text
from src.inference_utils import (
    list_trained_loras,
    load_inference_model,
    loaded_info,
    run_chat,
    unload_model,
)
from src.tb_utils import launch_tensorboard
from src.train_utils import TrainRequest, current_experiment, is_training, request_cancel, run_training

# 界面语言必须在 UI 构建前确定：Gradio 静态构建，启动参数 > 环境变量 > 中文默认
import sys as _sys


def _detect_lang() -> str:
    argv = _sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    return os.environ.get("UNSLOTH_GUI_LANG", "zh")


set_lang(_detect_lang())
_LAUNCH_ARGV: list[str] = list(_sys.argv)

MODELS, DATASETS, CONFIG_WARNINGS = safe_load_configs()
MODEL_DISPLAY_NAMES = [m.display_name for m in MODELS]


def _processed_names(items):
    return [d.display_name for d in items if getattr(d, "processed", False)]


# 训练 Tab 只认「数据处理」制成的统一数据；全部列表留给处理 Tab 做迁移来源
DATASET_DISPLAY_NAMES = _processed_names(DATASETS)
ALL_DATASET_NAMES = [d.display_name for d in DATASETS]


def _reload_datasets():
    """重读配置并刷新全局列表，返回告警信息."""
    global DATASET_DISPLAY_NAMES, ALL_DATASET_NAMES
    _, fresh, warns = safe_load_configs()
    DATASETS[:] = fresh
    DATASET_DISPLAY_NAMES = _processed_names(fresh)
    ALL_DATASET_NAMES = [d.display_name for d in fresh]
    return warns


MODEL_SOURCE_OPTIONS = [t("model.src.local"), t("model.src.hf"), t("model.src.ms")]


def _src_code(label: str) -> str:
    return {t("model.src.local"): "local", t("model.src.hf"): "huggingface",
            t("model.src.ms"): "modelscope"}.get(label or "", "huggingface")


def _src_label(code: str) -> str:
    return source_label("huggingface" if (code or "auto").lower() == "auto"
                        else code)


def _reload_models():
    """重读 models.json 并刷新全局列表，返回告警信息."""
    global MODEL_DISPLAY_NAMES
    try:
        MODELS[:] = load_models_config()
    except Exception as e:
        return [t("warn.models", err=e)]
    MODEL_DISPLAY_NAMES = [m.display_name for m in MODELS]
    return []


def _model_rows():
    rows = []
    for m in MODELS:
        rows.append([
            m.display_name,
            source_label(model_source(m)),
            m.model_id,
            t("common.yes") if m.load_in_4bit else t("common.no"),
            m.dtype or t("common.auto"),
            str(m.max_seq_length),
        ])
    return rows


def _model_dd_updates(cur_train, cur_inf):
    """训练/测试下拉刷新：保留仍有效的选择，失效则回落第一个."""
    train_val = (cur_train if cur_train in MODEL_DISPLAY_NAMES
                 else (MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None))
    inf_val = (cur_inf if cur_inf in MODEL_DISPLAY_NAMES
               else (MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None))
    return (
        gr.update(choices=MODEL_DISPLAY_NAMES, value=train_val),
        gr.update(choices=MODEL_DISPLAY_NAMES, value=inf_val),
    )


# --- 模型管理：主从编辑（点行填充表单，新建/保存/两步删除） ---
# 两步删除的待确认名（单机本地应用，模块级变量足够）
_del_armed: str | None = None


def _reset_del_arm():
    global _del_armed
    _del_armed = None


def _model_pick(evt: gr.SelectData):
    """点表格某行，把该模型填进表单."""
    _reset_del_arm()
    empty = [gr.update(value=v) for v in
             ("", t("model.src.hf"), "", True, t("common.auto"), 2048, "")]
    try:
        row_idx = evt.index[0] if evt is not None and evt.index else None
    except Exception:
        row_idx = None
    if row_idx is None or not (0 <= row_idx < len(MODELS)):
        return (*empty, t("m.pick_empty"))
    m = MODELS[row_idx]
    return (
        gr.update(value=m.display_name),
        gr.update(value=_src_label(model_source(m))),
        gr.update(value=m.model_id),
        gr.update(value=bool(m.load_in_4bit)),
        gr.update(value=m.dtype or t("common.auto")),
        gr.update(value=m.max_seq_length),
        gr.update(value=m.chat_template or ""),
        t("m.pick_loaded", name=m.display_name),
    )


def _model_new():
    """清空表单，准备录一个新模型."""
    _reset_del_arm()
    return (
        gr.update(value=""), gr.update(value=t("model.src.hf")), gr.update(value=""),
        gr.update(value=True), gr.update(value=t("common.auto")), gr.update(value=2048),
        gr.update(value=""),
        t("m.new_cleared"),
    )


def _model_save(display, source, model_id, use_4bit, dtype, seq_len, chat,
                cur_train, cur_inf):
    from src.config import find_by_name as _find
    _reset_del_arm()
    code = _src_code(source)
    display = (display or "").strip()
    model_id = (model_id or "").strip()
    if not display:
        return (t("m.err.noname"),) + (gr.update(),) * 3
    if not model_id:
        return (t("m.err.noid"),) + (gr.update(),) * 3
    if code != "local" and " " in model_id:
        return (t("m.err.space"),) + (gr.update(),) * 3
    if code == "local":
        p = Path(model_id)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            return (t("m.err.local_missing", id=model_id, path=p),) + (gr.update(),) * 3
    try:
        seq = int(seq_len)
    except (TypeError, ValueError):
        return (t("m.err.seq_nan"),) + (gr.update(),) * 3
    if seq < 256:
        return (t("m.err.seq_min"),) + (gr.update(),) * 3
    existed = _find(MODELS, display) is not None
    cfg = ModelConfig(
        display_name=display, model_id=model_id, load_in_4bit=bool(use_4bit),
        dtype=None if dtype == t("common.auto") else dtype, max_seq_length=seq,
        chat_template=(chat or "").strip() or None,
        source=code,
    )
    MODELS[:] = [c for c in MODELS if c.display_name != display] + [cfg]
    try:
        save_models_config(MODELS)
    except Exception as e:
        return (t("m.err.write", err=e),) + (gr.update(),) * 3
    warns = _reload_models()
    msg = t("m.saved", action=t("m.act_update") if existed else t("m.act_add"),
            display=display, source=source)
    if code == "modelscope":
        msg += t("m.ms_note")
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    dd_train, dd_inf = _model_dd_updates(cur_train, cur_inf)
    return (msg, gr.update(value=_model_rows()), dd_train, dd_inf)


def _model_delete_step(name, cur_train, cur_inf):
    """两步删除：目标就是表单里的展示名。第一次点进入待确认，第二次执行；
    期间点其他按钮自动取消。"""
    global _del_armed
    name = (name or "").strip()
    if not name:
        return (t("m.err.empty_form"),) + (gr.update(),) * 3
    if len(MODELS) <= 1:
        return (t("m.err.last"),) + (gr.update(),) * 3
    if _del_armed != name:
        _del_armed = name
        return (t("m.del_arm", name=name),) + (gr.update(),) * 3
    _del_armed = None
    MODELS[:] = [c for c in MODELS if c.display_name != name]
    try:
        save_models_config(MODELS)
    except Exception as e:
        return (t("m.err.write", err=e),) + (gr.update(),) * 3
    warns = _reload_models()
    msg = t("m.deleted", name=name)
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    dd_train, dd_inf = _model_dd_updates(cur_train, cur_inf)
    return (msg, gr.update(value=_model_rows()), dd_train, dd_inf)


def _models_refresh(cur_train, cur_inf):
    _reset_del_arm()
    warns = _reload_models()
    msg = t("m.refreshed", n=len(MODEL_DISPLAY_NAMES))
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    dd_train, dd_inf = _model_dd_updates(cur_train, cur_inf)
    return (msg, gr.update(value=_model_rows()), dd_train, dd_inf)


# --- 数据管理：主从查看 + 样本/重命名/两步删除 ---
_data_del_armed: str | None = None


def _data_rows():
    return [[e["name"], e["rows"], e["source"], e["created"], e["size"],
             ("✅" if e["trainable"] else "❌")] for e in list_entries()]


def _reset_data_arm():
    global _data_del_armed
    _data_del_armed = None


def _map_train_sel(cur, old=None, new=None):
    vals = list(cur or [])
    if old is not None:
        vals = [new if v == old else v for v in vals] if new else [
            v for v in vals if v != old]
    vals = [v for v in vals if v in DATASET_DISPLAY_NAMES]
    return gr.update(choices=DATASET_DISPLAY_NAMES, value=vals)


def _map_prep_sel(cur, old=None, new=None):
    if old is not None:
        cur = new if cur == old and new else (None if cur == old else cur)
    return gr.update(choices=ALL_DATASET_NAMES,
                     value=cur if cur in ALL_DATASET_NAMES else (
                         ALL_DATASET_NAMES[0] if ALL_DATASET_NAMES else None))


def _data_pick(evt: gr.SelectData):
    _reset_data_arm()
    try:
        idx = evt.index[0] if evt is not None and evt.index else None
    except Exception:
        idx = None
    entries = list_entries()
    if idx is None or not (0 <= idx < len(entries)):
        return "", t("d.pick_empty")
    name = entries[idx]["name"]
    return name, detail_text(name)


def _data_sample(name):
    if not name:
        return t("d.err.noselect_sample")
    return sample_text(name)


def _data_delete_step(name, cur_train, cur_prep):
    global _data_del_armed
    if not name:
        return (t("d.err.noselect"),) + (gr.update(),) * 5
    if _data_del_armed != name:
        _data_del_armed = name
        return (t("d.del_arm", name=name),) + (gr.update(),) * 5
    _data_del_armed = None
    msg = delete_entry(name)
    if msg.startswith("❌"):
        return (msg,) + (gr.update(),) * 5
    _reload_datasets()
    return (msg, gr.update(value=_data_rows()), "", t("d.deleted_detail"),
            _map_train_sel(cur_train, old=name),
            _map_prep_sel(cur_prep, old=name))


def _data_rename(name, new_name, cur_train, cur_prep):
    _reset_data_arm()
    try:
        new = rename_entry(name or "", new_name or "")
    except ValueError as e:
        return (t("d.err.rename", err=e),) + (gr.update(),) * 5
    _reload_datasets()
    return (t("d.renamed", new=new),
            gr.update(value=_data_rows()), new, detail_text(new),
            _map_train_sel(cur_train, old=name, new=new),
            _map_prep_sel(cur_prep, old=name, new=new))


def _data_refresh(cur_train, cur_prep, selected):
    _reset_data_arm()
    warns = _reload_datasets()
    names = [e["name"] for e in list_entries()]
    if selected and selected in names:
        detail, sel = detail_text(selected), selected
    else:
        detail, sel = t("d.detail_empty"), ""
    msg = t("d.refreshed", n=len(names))
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    return (msg, gr.update(value=_data_rows()), sel, detail,
            _map_train_sel(cur_train), _map_prep_sel(cur_prep))

# --- 纯视觉主题（不影响任何功能逻辑） ---
APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
)

APP_CSS = """
footer {display: none !important}
body {font-family: -apple-system, "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif !important}
#app-hero {
  background: linear-gradient(135deg, #312e81 0%, #4f46e5 55%, #0284c7 100%);
  border-radius: 16px; padding: 28px 32px; margin-bottom: 12px; color: #fff;
  box-shadow: 0 8px 24px rgba(49, 46, 129, .25);
}
#app-hero h1 {margin: 0; font-size: 28px; letter-spacing: .5px}
#app-hero p {margin: 8px 0 0; opacity: .85; font-size: 14px}
#app-hero .ver {
  display: inline-block; margin-top: 10px; font-size: 12px;
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  padding: 2px 10px; border-radius: 999px;
}
button.lg.primary {font-weight: 600}
.tabs > .tab-nav button {font-weight: 500}
"""

HERO_HTML = f"""
<div id="app-hero">
  <h1>{t("hero.title")}</h1>
  <p>{t("hero.subtitle")}</p>
  <span class="ver">{t("hero.ver")}</span>
</div>
"""


def _update_training_mode_ui(mode):
    is_epoch_mode = mode == "按轮次 (Epochs)"
    return (
        gr.update(visible=is_epoch_mode),
        gr.update(visible=not is_epoch_mode),
        gr.update(visible=not is_epoch_mode),
    )


def _on_dataset_change(selected: list[str]):
    """选中数据集后：显示预览 + 自动套用 recommended_params（如孙悟空示例）."""
    if not selected:
        return t("pd.none"), {}
    parts = []
    merged_params: dict = {}
    for name in selected:
        cfg = find_by_name(DATASETS, name)
        if cfg is None:
            parts.append(t("pd.missing", name=name))
            continue
        parts.append(dataset_preview_text(cfg, n=2))
        if cfg.recommended_params:
            merged_params.update(cfg.recommended_params)
    updates = {}
    if merged_params:
        # 仅回填 UI 不强制覆盖用户手调：这里直接给出建议值更新
        if "training_mode" in merged_params:
            updates["training_mode"] = merged_params["training_mode"]
        for k in ("num_epochs", "lora_r", "lora_alpha", "learning_rate"):
            if k in merged_params:
                updates[k] = merged_params[k]
    tip = ""
    if updates:
        tip = t("pd.reco", updates=updates)
    return "\n\n".join(parts) + tip, updates


def _apply_recommended(updates: dict, cur_mode, cur_epoch, cur_r, cur_alpha, cur_lr):
    if not updates:
        return cur_mode, cur_epoch, cur_r, cur_alpha, cur_lr
    mode = updates.get("training_mode", cur_mode)
    return (
        mode,
        float(updates.get("num_epochs", cur_epoch)),
        int(updates.get("lora_r", cur_r)),
        int(updates.get("lora_alpha", cur_alpha)),
        float(updates.get("learning_rate", cur_lr)),
    )


def _train_wrapper(experiment_name, resume_training, truncate_dataset, max_samples,
                   training_mode, num_epochs, max_steps, save_steps,
                   selected_model_name, selected_dataset_names,
                   lora_r, lora_alpha, batch_size, grad_accum, lr, max_seq_length,
                   progress=gr.Progress(track_tqdm=True)):
    try:
        req = TrainRequest(
            experiment_name=experiment_name, resume_training=bool(resume_training),
            truncate_dataset=bool(truncate_dataset), max_samples=int(max_samples or 200),
            training_mode=training_mode, num_epochs=float(num_epochs),
            max_steps=int(max_steps or 100), save_steps=int(save_steps or 50),
            selected_model_name=selected_model_name,
            selected_dataset_names=list(selected_dataset_names or []),
            lora_r=int(lora_r), lora_alpha=int(lora_alpha),
            batch_size=int(batch_size), grad_accum=int(grad_accum),
            lr=float(lr), max_seq_length=int(max_seq_length),
        )
    except (TypeError, ValueError) as e:
        yield t("t.err.params", err=e), gr.skip()
        return
    last = ""
    for msg in run_training(req, progress=progress):
        last = msg
        yield last, gr.skip()
    # 训练结束刷新测试 Tab 的 LoRA 下拉
    yield last, gr.update(choices=list_trained_loras())


def _stop_training():
    return request_cancel()


def _load_model_wrapper(base, lora, progress=gr.Progress(track_tqdm=True)):
    msg = load_inference_model(base, lora, progress=progress)
    return f"{msg}\n{loaded_info()}", gr.update(choices=list_trained_loras())


# --- 数据处理 Tab 逻辑 ---
def _prep_source_ui(kind):
    is_upload = kind == t("p.src.upload")
    is_hf = kind == t("p.src.hf")
    is_ex = kind == t("p.src.existing")
    return (
        gr.update(visible=is_upload),
        gr.update(visible=is_hf),
        gr.update(visible=is_hf),
        gr.update(visible=is_ex),
        # 换来源后之前的检查结果作废，强制重读
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _prep_inspect(kind, upload_path, hf_id, hf_split, existing_name,
                  progress=gr.Progress(track_tqdm=True)):
    from src.config import find_by_name as _find
    _empty = (gr.update(), gr.update(), gr.update(), gr.update())
    _locked = (gr.update(interactive=False), gr.update(interactive=False))
    if kind == t("p.src.upload"):
        if not upload_path:
            return t("p.err.no_file"), {}, *_empty, *_locked
        try:
            local_path = persist_upload(upload_path)
        except Exception as e:
            return t("p.err.save_fail", err=e), {}, *_empty, *_locked
        dkind, existing = "upload", None
    elif kind == t("p.src.hf"):
        dkind, local_path, existing = "hf", "", None
    else:
        dkind, local_path = "existing", ""
        existing = _find(DATASETS, existing_name)
        if existing is None:
            return t("p.err.no_config", name=existing_name), {}, *_empty, *_locked
    try:
        text, state = inspect_text(dkind, hf_id or "", hf_split or "train", local_path, existing)
    except Exception as e:
        return t("p.err.read_fail", err=e), {}, *_empty, *_locked
    cols = state["columns"]
    return (
        text, state,
        gr.update(choices=cols, value=[cols[0]] if cols else []),
        gr.update(choices=cols, value=[]),
        gr.update(choices=cols, value=[]),
        gr.update(choices=cols, value=[cols[-1]] if cols else []),
        # 检查通过才放行预览和生成
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _prep_preview_row(state, ins_cols, input_cols, think_cols, out_cols, fixed_ins):
    try:
        return preview_row(state or {}, ins_cols, input_cols, think_cols, out_cols,
                           fixed_ins or "")
    except Exception as e:
        return t("dp.preview.fail", err=e)


def _prep_generate(state, ins_cols, input_cols, think_cols, out_cols, fixed_ins, name,
                   progress=gr.Progress(track_tqdm=True)):
    try:
        status, preview, new_name = generate_unified(
            state or {}, ins_cols, input_cols, think_cols, out_cols, name,
            fixed_instruction=fixed_ins or "", progress=progress)
    except Exception as e:
        return t("dp.err.gen_fail", err=e), "", gr.update(), gr.update(), gr.update()
    warns = _reload_datasets()
    msg = status
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    return (
        msg, preview,
        gr.update(choices=DATASET_DISPLAY_NAMES, value=[new_name]),
        gr.update(choices=ALL_DATASET_NAMES),
        gr.update(value=_data_rows()),
    )


def _refresh_train_datasets():
    warns = _reload_datasets()
    msg = t("t.refreshed", n=len(DATASET_DISPLAY_NAMES))
    if warns:
        msg += t("common.warns", warns="; ".join(warns))
    if not DATASET_DISPLAY_NAMES:
        msg += t("t.no_data")
    return msg, gr.update(choices=DATASET_DISPLAY_NAMES)


def _switch_lang(choice):
    """界面语言切换：Gradio 静态构建，只能重启进程生效。"""
    lang = "en" if (choice or "") == "English" else "zh"
    if lang == get_lang():
        return gr.skip()
    script = cleaned_script()
    new_argv = [script] + cleaned_args() + ["--lang", lang]
    import sys as _s
    try:
        os.execv(_s.executable, [_s.executable] + new_argv)
    except Exception as e:
        gr.Warning(t("lang.restart_fail", cmd="python " + " ".join(new_argv), err=e))
    return gr.skip()


def cleaned_script() -> str:
    a0 = _LAUNCH_ARGV[0] if _LAUNCH_ARGV else "app.py"
    return a0 if a0.endswith(".py") else str(PROJECT_ROOT / "app.py")


def cleaned_args() -> list:
    out, skip_next = [], False
    for a in _LAUNCH_ARGV[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--lang":
            skip_next = True
            continue
        if a.startswith("--lang="):
            continue
        out.append(a)
    return out


with gr.Blocks() as demo:
    gr.HTML(HERO_HTML)
    with gr.Row():
        lang_radio = gr.Radio(["中文", "English"],
                              value="English" if get_lang() == "en" else "中文",
                              label=t("lang.label"), scale=1)
        gr.Markdown(t("lang.hint"), scale=4)
    if CONFIG_WARNINGS:
        gr.Markdown("⚠️ " + "\n\n⚠️ ".join(CONFIG_WARNINGS))
    if is_training():
        gr.Markdown(t("app.training_running", exp=current_experiment()))

    with gr.Tabs():
        with gr.Tab(t("tab.models")):
            gr.Markdown(t("m.title"))
            gr.Markdown(t("m.subtitle"))
            with gr.Row():
                with gr.Column(scale=3):
                    models_table = gr.Dataframe(
                        headers=[t("m.col.name"), t("m.col.source"), t("m.col.id"),
                                 t("m.col.bit"), t("m.col.dtype"), t("m.col.seq")],
                        datatype=["str"] * 6, row_count=(0, "dynamic"), column_count=6,
                        interactive=False, wrap=True, value=_model_rows(),
                    )
                    m_refresh_btn = gr.Button(t("m.refresh"), size="sm")
                with gr.Column(scale=2):
                    m_display = gr.Textbox(label=t("m.display"),
                                           placeholder=t("m.display_ph"))
                    m_source = gr.Radio(MODEL_SOURCE_OPTIONS, value=t("model.src.hf"),
                                        label=t("m.source"))
                    m_id = gr.Textbox(
                        label=t("m.id"),
                        placeholder=t("m.id_ph"),
                        lines=2,
                    )
                    with gr.Row():
                        m_4bit = gr.Checkbox(label=t("m.bit"), value=True)
                        m_dtype = gr.Dropdown(label=t("m.dtype"),
                                              choices=[t("common.auto"), "bfloat16", "float16"],
                                              value=t("common.auto"))
                    with gr.Row():
                        m_seq = gr.Number(label=t("m.seq"), value=2048,
                                          minimum=256, step=256)
                        m_chat = gr.Textbox(label=t("m.chat"),
                                            placeholder=t("m.chat_ph"))
                    with gr.Row():
                        m_new_btn = gr.Button(t("m.new"), size="sm")
                        m_save_btn = gr.Button(t("m.save"), variant="primary")
                        m_del_btn = gr.Button(t("m.del"), variant="stop")
                    m_status = gr.Textbox(label=t("m.status"), interactive=False,
                                          lines=3, max_lines=8)
                    gr.Markdown(
                        t("m.hint"),
                        elem_classes=["hint"],
                    )

        with gr.Tab(t("tab.data")):
            gr.Markdown(t("d.title"))
            gr.Markdown(t("d.subtitle"))
            with gr.Row():
                with gr.Column(scale=2):
                    data_table = gr.Dataframe(
                        headers=[t("d.col.name"), t("d.col.rows"), t("d.col.source"),
                                 t("d.col.created"), t("d.col.size"), t("d.col.trainable")],
                        datatype=["str"] * 6, row_count=(0, "dynamic"), column_count=6,
                        interactive=False, wrap=True, value=_data_rows(),
                    )
                    d_refresh_btn = gr.Button(t("d.refresh"), size="sm")
                with gr.Column(scale=3):
                    d_detail = gr.Textbox(label=t("d.detail"), interactive=False,
                                          lines=10, max_lines=20,
                                          value=t("d.detail_empty"))
                    d_sample = gr.Textbox(label=t("d.sample"),
                                          interactive=False, lines=10, max_lines=20)
                    d_new_name = gr.Textbox(label=t("d.newname"),
                                            placeholder=t("d.newname_ph"))
                    with gr.Row():
                        d_sample_btn = gr.Button(t("d.sample_btn"), variant="secondary", size="sm")
                        d_rename_btn = gr.Button(t("d.rename"), size="sm")
                        d_del_btn = gr.Button(t("d.del"), variant="stop", size="sm")
                    d_status = gr.Textbox(label=t("d.status"), interactive=False,
                                          lines=3, max_lines=8)
            d_selected = gr.State("")

        with gr.Tab(t("tab.prep")):
            gr.Markdown(t("p.title"))
            gr.Markdown(t("p.steps"))
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion(t("p.src_acc"), open=True):
                        prep_source_radio = gr.Radio(
                            [t("p.src.upload"), t("p.src.hf"), t("p.src.existing")],
                            value=t("p.src.upload"), label=t("p.src_label"),
                        )
                        prep_upload = gr.File(
                            label=t("p.upload"),
                            file_types=[".jsonl", ".json", ".csv", ".parquet", ".txt"],
                            type="filepath", visible=True,
                        )
                        prep_hf_id = gr.Textbox(
                            label=t("p.hf_id"), value="yahma/alpaca-cleaned", visible=False,
                        )
                        prep_hf_split = gr.Textbox(label=t("p.hf_split"), value="train", visible=False)
                        prep_existing = gr.Dropdown(
                            label=t("p.existing"),
                            choices=ALL_DATASET_NAMES,
                            value=ALL_DATASET_NAMES[0] if ALL_DATASET_NAMES else None,
                            visible=False,
                        )
                        prep_inspect_btn = gr.Button(t("p.inspect_btn"), variant="secondary")
                    with gr.Accordion(t("p.map_acc"), open=True):
                        gr.Markdown(t("p.map_hint"))
                        prep_ins_col = gr.Dropdown(
                            label=t("p.ins_col"),
                            choices=[], multiselect=True,
                        )
                        prep_fixed_ins = gr.Textbox(
                            label=t("p.fixed"),
                            placeholder=t("p.fixed_ph"),
                            lines=3,
                        )
                        prep_input_col = gr.Dropdown(
                            label=t("p.input_col"),
                            choices=[], multiselect=True,
                        )
                        prep_think_col = gr.Dropdown(
                            label=t("p.think_col"),
                            choices=[], multiselect=True,
                        )
                        prep_out_col = gr.Dropdown(
                            label=t("p.out_col"),
                            choices=[], multiselect=True,
                        )
                        prep_name = gr.Textbox(
                            label=t("p.name"),
                            placeholder=t("p.name_ph"),
                        )
                        with gr.Row():
                            prep_preview_btn = gr.Button(t("p.preview_btn"), variant="secondary",
                                                         interactive=False)
                            prep_generate_btn = gr.Button(t("p.generate_btn"), variant="primary",
                                                          interactive=False)
                with gr.Column(scale=2):
                    prep_inspect_output = gr.Textbox(
                        label=t("p.inspect_out"), interactive=False, lines=12, max_lines=25,
                    )
                    prep_status = gr.Textbox(label=t("p.status"), interactive=False, lines=3, max_lines=8)
                    prep_preview = gr.Textbox(
                        label=t("p.preview"), interactive=False, lines=12, max_lines=25,
                    )
            prep_state = gr.State({})

        with gr.Tab(t("tab.train")):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(t("t.cfg"))
                    with gr.Accordion(t("t.exp"), open=True):
                        experiment_name_input = gr.Textbox(label=t("t.exp_name"), value="8gb-vram-test")
                        resume_checkbox = gr.Checkbox(label=t("t.resume"), value=False)
                    with gr.Accordion(t("t.model_data"), open=True):
                        model_dropdown = gr.Dropdown(
                            choices=MODEL_DISPLAY_NAMES,
                            value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                            label=t("t.model"),
                        )
                        gr.Markdown(t("t.ds_hint"))
                        dataset_dropdown = gr.Dropdown(
                            choices=DATASET_DISPLAY_NAMES,
                            value=(
                                [DATASET_DISPLAY_NAMES[0]]
                                if DATASET_DISPLAY_NAMES
                                else []
                            ),
                            label=t("t.ds"),
                            multiselect=True,
                        )
                        with gr.Row():
                            refresh_datasets_btn = gr.Button(t("t.refresh"), size="sm")
                            preview_btn = gr.Button(t("t.preview_ds"), size="sm")
                        truncate_dataset_checkbox = gr.Checkbox(
                            label=t("t.truncate"),
                            value=True,
                            info=t("t.truncate_info"),
                        )
                        max_samples_input = gr.Number(
                            value=200, label=t("t.max_samples"), minimum=10, maximum=100000, step=10,
                        )
                    with gr.Accordion(t("t.lora"), open=False):
                        lora_r_slider = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank (r)")
                        lora_alpha_slider = gr.Slider(4, 128, value=16, step=4, label="LoRA Alpha")
                    with gr.Accordion(t("t.core"), open=True):
                        training_mode_selector = gr.Radio(
                            ["按步数 (Steps)", "按轮次 (Epochs)"],
                            value="按步数 (Steps)", label="训练模式",
                        )
                        num_epochs_slider = gr.Slider(0.1, 10, value=1, step=0.1,
                                                     label=t("t.epochs"), visible=False)
                        max_steps_slider = gr.Slider(10, 2000, value=100, step=10,
                                                    label=t("t.steps"), visible=True)
                        save_steps_input = gr.Number(value=50, label=t("t.save_steps"), visible=True)
                        batch_size_slider = gr.Slider(1, 16, value=1, step=1, label=t("t.batch"))
                        grad_accum_slider = gr.Slider(1, 16, value=8, step=1, label=t("t.grad"))
                        learning_rate_slider = gr.Slider(1e-5, 5e-4, value=2e-4, step=1e-5, label=t("t.lr"))
                        max_seq_len_slider = gr.Slider(512, 8192, value=2048, step=256, label=t("t.seq"))
                    with gr.Row():
                        start_button = gr.Button(t("t.start"), variant="primary")
                        stop_button = gr.Button(t("t.stop"), variant="stop")
                with gr.Column(scale=3):
                    gr.Markdown(t("t.tb"))
                    tb_status = gr.Textbox(label=t("t.tb_status"), interactive=False)
                    tensorboard_view = gr.HTML(t("t.tb_empty"))
            gr.Markdown(t("t.preview_head"))
            preview_output = gr.Textbox(label=t("t.preview_label"), interactive=False,
                                        lines=8, max_lines=20)
            gr.Markdown(t("t.status_head"))
            status_output = gr.Textbox(label=t("t.status_label"), interactive=False,
                                       lines=5, max_lines=20)

        with gr.Tab(t("tab.test")):
            gr.Markdown(t("i.title"))
            gr.Markdown(t("i.steps"))
            with gr.Row():
                inference_model_selector = gr.Dropdown(
                    label=t("i.model"),
                    choices=MODEL_DISPLAY_NAMES,
                    value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                )
                lora_selector_dropdown = gr.Dropdown(
                    label=t("i.lora"), choices=list_trained_loras()
                )
            with gr.Row():
                load_model_button = gr.Button(t("i.load"), variant="primary")
                unload_model_button = gr.Button(t("i.unload"))
                refresh_lora_btn = gr.Button(t("i.refresh"), size="sm")
            load_status_textbox = gr.Textbox(label=t("i.load_status"), interactive=False)
            system_prompt_textbox = gr.Textbox(
                label=t("i.system"),
                info=t("i.system_info"),
                lines=3,
                value=t("i.system_default"),
            )
            with gr.Accordion(t("i.gen"), open=False):
                max_tokens_slider = gr.Slider(32, 2048, value=256, step=32, label=t("i.max_tokens"))
                temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label=t("i.temp"))
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label=t("i.top_p"))
                top_k_slider = gr.Slider(1, 100, value=40, step=1, label=t("i.top_k"))
            chatbot = gr.Chatbot(label=t("i.chat"), height=500)
            with gr.Row():
                chat_input_textbox = gr.Textbox(
                    show_label=False, placeholder=t("i.input_ph"), scale=4, container=False,
                )
                submit_button = gr.Button(t("i.send"), variant="primary", scale=1)
                clear_button = gr.Button(t("i.clear"), scale=1)

    # --- 事件绑定 ---
    # 数据管理 Tab：主从查看
    data_table.select(
        fn=_data_pick,
        outputs=[d_selected, d_detail],
    )
    d_sample_btn.click(
        fn=_data_sample,
        inputs=[d_selected],
        outputs=[d_sample],
    )
    d_del_btn.click(
        fn=_data_delete_step,
        inputs=[d_selected, dataset_dropdown, prep_existing],
        outputs=[d_status, data_table, d_selected, d_detail,
                 dataset_dropdown, prep_existing],
    )
    d_rename_btn.click(
        fn=_data_rename,
        inputs=[d_selected, d_new_name, dataset_dropdown, prep_existing],
        outputs=[d_status, data_table, d_selected, d_detail,
                 dataset_dropdown, prep_existing],
    )
    d_refresh_btn.click(
        fn=_data_refresh,
        inputs=[dataset_dropdown, prep_existing, d_selected],
        outputs=[d_status, data_table, d_selected, d_detail,
                 dataset_dropdown, prep_existing],
    )

    # 模型管理 Tab：主从编辑
    models_table.select(
        fn=_model_pick,
        outputs=[m_display, m_source, m_id, m_4bit, m_dtype, m_seq, m_chat, m_status],
    )
    m_new_btn.click(
        fn=_model_new,
        outputs=[m_display, m_source, m_id, m_4bit, m_dtype, m_seq, m_chat, m_status],
    )
    m_save_btn.click(
        fn=_model_save,
        inputs=[m_display, m_source, m_id, m_4bit, m_dtype, m_seq, m_chat,
                model_dropdown, inference_model_selector],
        outputs=[m_status, models_table, model_dropdown, inference_model_selector],
    )
    m_del_btn.click(
        fn=_model_delete_step,
        inputs=[m_display, model_dropdown, inference_model_selector],
        outputs=[m_status, models_table, model_dropdown, inference_model_selector],
    )
    m_refresh_btn.click(
        fn=_models_refresh,
        inputs=[model_dropdown, inference_model_selector],
        outputs=[m_status, models_table, model_dropdown, inference_model_selector],
    )

    # 数据处理 Tab
    prep_source_radio.change(
        fn=_prep_source_ui,
        inputs=[prep_source_radio],
        outputs=[prep_upload, prep_hf_id, prep_hf_split, prep_existing,
                 prep_preview_btn, prep_generate_btn],
    )
    prep_inspect_btn.click(
        fn=_prep_inspect,
        inputs=[prep_source_radio, prep_upload, prep_hf_id, prep_hf_split, prep_existing],
        outputs=[prep_inspect_output, prep_state, prep_ins_col, prep_input_col,
                 prep_think_col, prep_out_col, prep_preview_btn, prep_generate_btn],
    )
    prep_preview_btn.click(
        fn=_prep_preview_row,
        inputs=[prep_state, prep_ins_col, prep_input_col, prep_think_col, prep_out_col,
                prep_fixed_ins],
        outputs=[prep_preview],
    )
    prep_generate_btn.click(
        fn=_prep_generate,
        inputs=[prep_state, prep_ins_col, prep_input_col, prep_think_col, prep_out_col,
                prep_fixed_ins, prep_name],
        outputs=[prep_status, prep_preview, dataset_dropdown, prep_existing, data_table],
    )
    refresh_datasets_btn.click(
        fn=_refresh_train_datasets,
        outputs=[preview_output, dataset_dropdown],
    )

    training_mode_selector.change(
        fn=_update_training_mode_ui,
        inputs=training_mode_selector,
        outputs=[num_epochs_slider, max_steps_slider, save_steps_input],
    )

    _reco_state = gr.State({})

    def _on_dataset_change_wrap(selected):
        text, updates = _on_dataset_change(selected)
        return text, updates

    dataset_dropdown.change(
        fn=_on_dataset_change_wrap,
        inputs=[dataset_dropdown],
        outputs=[preview_output, _reco_state],
    ).then(
        fn=_apply_recommended,
        inputs=[_reco_state, training_mode_selector, num_epochs_slider,
                lora_r_slider, lora_alpha_slider, learning_rate_slider],
        outputs=[training_mode_selector, num_epochs_slider, lora_r_slider,
                 lora_alpha_slider, learning_rate_slider],
    ).then(
        # 推荐参数可能切换了训练模式，联动滑块显隐
        fn=_update_training_mode_ui,
        inputs=[training_mode_selector],
        outputs=[num_epochs_slider, max_steps_slider, save_steps_input],
    )
    preview_btn.click(
        fn=lambda sel: _on_dataset_change(sel or [])[0],
        inputs=[dataset_dropdown], outputs=[preview_output],
    )

    train_inputs = [
        experiment_name_input, resume_checkbox, truncate_dataset_checkbox, max_samples_input,
        training_mode_selector, num_epochs_slider, max_steps_slider, save_steps_input,
        model_dropdown, dataset_dropdown, lora_r_slider, lora_alpha_slider,
        batch_size_slider, grad_accum_slider, learning_rate_slider, max_seq_len_slider,
    ]
    start_button.click(fn=_train_wrapper, inputs=train_inputs,
                       outputs=[status_output, lora_selector_dropdown])
    stop_button.click(fn=_stop_training, outputs=[status_output])

    load_model_button.click(
        fn=_load_model_wrapper,
        inputs=[inference_model_selector, lora_selector_dropdown],
        outputs=[load_status_textbox, lora_selector_dropdown],
    )
    unload_model_button.click(fn=lambda: f"{unload_model()}\n{loaded_info()}",
                              outputs=[load_status_textbox])
    refresh_lora_btn.click(fn=lambda: gr.update(choices=list_trained_loras()),
                           outputs=[lora_selector_dropdown])

    def _chat_wrap(user_input, history, system_prompt, max_tokens, temp, top_p, top_k):
        yield from run_chat(user_input, history, system_prompt,
                            max_new_tokens=max_tokens, temperature=temp,
                            top_p=top_p, top_k=top_k)

    chat_inputs = [chat_input_textbox, chatbot, system_prompt_textbox,
                   max_tokens_slider, temp_slider, top_p_slider, top_k_slider]
    submit_event = chat_input_textbox.submit(fn=_chat_wrap, inputs=chat_inputs, outputs=[chatbot])
    submit_event.then(lambda: gr.update(value=""), outputs=[chat_input_textbox])
    button_event = submit_button.click(fn=_chat_wrap, inputs=chat_inputs, outputs=[chatbot])
    button_event.then(lambda: gr.update(value=""), outputs=[chat_input_textbox])
    clear_button.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input_textbox])

    def _on_load(tb_port):
        ok, msg, port = launch_tensorboard(port=int(tb_port))
        html = (f'<iframe src="http://127.0.0.1:{port}" width="100%" height="800px" frameborder="0"></iframe>'
                if ok else f"<p>{msg}</p>")
        return msg, html

    tb_port_state = gr.State(6006)
    demo.load(fn=_on_load, inputs=[tb_port_state], outputs=[tb_status, tensorboard_view])

    lang_radio.change(fn=_switch_lang, inputs=[lang_radio], outputs=[lang_radio])


def main():
    parser = argparse.ArgumentParser(description="Unsloth GUI Trainer v0.1.0")
    parser.add_argument("--host", default=os.environ.get("GRADIO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE", "") == "1",
                        help="是否生成公网 share 链接（默认关闭，安全）")
    parser.add_argument("--tb-port", type=int, default=int(os.environ.get("TB_PORT", "6006")))
    parser.add_argument("--lang", choices=["zh", "en"], default=None,
                        help="界面语言 zh/en（默认中文，也可用环境变量 UNSLOTH_GUI_LANG）")
    args = parser.parse_args()
    if args.lang:
        set_lang(args.lang)

    ok, msg, port = launch_tensorboard(port=args.tb_port)
    print(msg)

    demo.queue(max_size=8).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        inbrowser=False, theme=APP_THEME, css=APP_CSS,
    )


if __name__ == "__main__":
    main()
