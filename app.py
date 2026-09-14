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
from src.config import SOURCE_LABEL
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


MODEL_SOURCE_OPTIONS = ["本地路径", "HuggingFace", "魔搭 ModelScope"]
_MODEL_SOURCE_MAP = {"本地路径": "local", "HuggingFace": "huggingface",
                     "魔搭 ModelScope": "modelscope"}


def _reload_models():
    """重读 models.json 并刷新全局列表，返回告警信息."""
    global MODEL_DISPLAY_NAMES
    try:
        MODELS[:] = load_models_config()
    except Exception as e:
        return [f"模型配置加载失败: {e}"]
    MODEL_DISPLAY_NAMES = [m.display_name for m in MODELS]
    return []


def _model_rows():
    rows = []
    for m in MODELS:
        src = model_source(m)
        rows.append([
            m.display_name,
            SOURCE_LABEL.get(src, src),
            m.model_id,
            "是" if m.load_in_4bit else "否",
            m.dtype or "自动",
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
_SOURCE_LABEL_REV = {"local": "本地路径", "huggingface": "HuggingFace",
                     "modelscope": "魔搭 ModelScope"}


def _reset_del_arm():
    global _del_armed
    _del_armed = None


def _model_pick(evt: gr.SelectData):
    """点表格某行，把该模型填进表单."""
    _reset_del_arm()
    empty = [gr.update(value=v) for v in ("", "HuggingFace", "", True, "自动", 2048, "")]
    try:
        row_idx = evt.index[0] if evt is not None and evt.index else None
    except Exception:
        row_idx = None
    if row_idx is None or not (0 <= row_idx < len(MODELS)):
        return (*empty, "❌ 请点击表格中的某一行。")
    m = MODELS[row_idx]
    src = model_source(m)
    return (
        gr.update(value=m.display_name),
        gr.update(value=_SOURCE_LABEL_REV.get(src, "HuggingFace")),
        gr.update(value=m.model_id),
        gr.update(value=bool(m.load_in_4bit)),
        gr.update(value=m.dtype or "自动"),
        gr.update(value=m.max_seq_length),
        gr.update(value=m.chat_template or ""),
        f"已载入 '{m.display_name}'，改完点保存，删点两次删除。",
    )


def _model_new():
    """清空表单，准备录一个新模型."""
    _reset_del_arm()
    return (
        gr.update(value=""), gr.update(value="HuggingFace"), gr.update(value=""),
        gr.update(value=True), gr.update(value="自动"), gr.update(value=2048),
        gr.update(value=""),
        "已清空，填完点保存即可新增（展示名已存在则覆盖更新）。",
    )


def _model_save(display, source, model_id, use_4bit, dtype, seq_len, chat,
                cur_train, cur_inf):
    from src.config import find_by_name as _find
    _reset_del_arm()
    display = (display or "").strip()
    model_id = (model_id or "").strip()
    if not display:
        return ("❌ 展示名不能为空。",) + (gr.update(),) * 3
    if not model_id:
        return ("❌ 模型 ID / 路径不能为空。",) + (gr.update(),) * 3
    if source != "本地路径" and " " in model_id:
        return ("❌ 远端模型 ID 不能包含空格，请检查。",) + (gr.update(),) * 3
    if source == "本地路径":
        p = Path(model_id)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            return (f"❌ 本地路径不存在: {model_id}（解析为 {p}）。"
                     f"请检查路径或换来源。",) + (gr.update(),) * 3
    try:
        seq = int(seq_len)
    except (TypeError, ValueError):
        return ("❌ 上下文长度必须是数字。",) + (gr.update(),) * 3
    if seq < 256:
        return ("❌ 上下文长度至少 256。",) + (gr.update(),) * 3
    existed = _find(MODELS, display) is not None
    cfg = ModelConfig(
        display_name=display, model_id=model_id, load_in_4bit=bool(use_4bit),
        dtype=None if dtype == "自动" else dtype, max_seq_length=seq,
        chat_template=(chat or "").strip() or None,
        source=_MODEL_SOURCE_MAP[source],
    )
    MODELS[:] = [c for c in MODELS if c.display_name != display] + [cfg]
    try:
        save_models_config(MODELS)
    except Exception as e:
        return (f"❌ 写 models.json 失败: {e}",) + (gr.update(),) * 3
    warns = _reload_models()
    action = "已更新" if existed else "已添加"
    msg = f"✅ {action}模型 '{display}'（{source}）。"
    if source == "魔搭 ModelScope":
        msg += "首次训练/加载时自动从魔搭下载到本地缓存（需 pip install modelscope）。"
    if warns:
        msg += f"\n⚠️ {warns}"
    dd_train, dd_inf = _model_dd_updates(cur_train, cur_inf)
    return (msg, gr.update(value=_model_rows()), dd_train, dd_inf)


def _model_delete_step(name, cur_train, cur_inf):
    """两步删除：目标就是表单里的展示名。第一次点进入待确认，第二次执行；
    期间点其他按钮自动取消。"""
    global _del_armed
    name = (name or "").strip()
    if not name:
        return ("❌ 表单展示名是空的：先点表格某一行，再删。",) + (gr.update(),) * 3
    if len(MODELS) <= 1:
        return ("❌ 至少保留一个模型，不能全删。",) + (gr.update(),) * 3
    if _del_armed != name:
        _del_armed = name
        return (f"⚠️ 再点一次「删除」确认删除 '{name}'。点其他按钮自动取消。",
                ) + (gr.update(),) * 3
    _del_armed = None
    MODELS[:] = [c for c in MODELS if c.display_name != name]
    try:
        save_models_config(MODELS)
    except Exception as e:
        return (f"❌ 写 models.json 失败: {e}",) + (gr.update(),) * 3
    warns = _reload_models()
    msg = f"🗑 已删除模型 '{name}'。" + (f"\n⚠️ {warns}" if warns else "")
    dd_train, dd_inf = _model_dd_updates(cur_train, cur_inf)
    return (msg, gr.update(value=_model_rows()), dd_train, dd_inf)


def _models_refresh(cur_train, cur_inf):
    _reset_del_arm()
    warns = _reload_models()
    msg = (f"已刷新：共 {len(MODEL_DISPLAY_NAMES)} 个模型。"
           + (f"\n⚠️ {warns}" if warns else ""))
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
        return "", "❌ 请点击表格中的某一行。"
    name = entries[idx]["name"]
    return name, detail_text(name)


def _data_sample(name):
    if not name:
        return "❌ 先在左边点选一行，再看样本。"
    return sample_text(name)


def _data_delete_step(name, cur_train, cur_prep):
    global _data_del_armed
    if not name:
        return ("❌ 先在左边点选一行。",) + (gr.update(),) * 5
    if _data_del_armed != name:
        _data_del_armed = name
        return (f"⚠️ 再点一次「删除」确认删除 '{name}'。点其他按钮自动取消。",
                ) + (gr.update(),) * 5
    _data_del_armed = None
    msg = delete_entry(name)
    if msg.startswith("❌"):
        return (msg,) + (gr.update(),) * 5
    _reload_datasets()
    return (msg, gr.update(value=_data_rows()), "", "已删除，点其他行查看。",
            _map_train_sel(cur_train, old=name),
            _map_prep_sel(cur_prep, old=name))


def _data_rename(name, new_name, cur_train, cur_prep):
    _reset_data_arm()
    try:
        new = rename_entry(name or "", new_name or "")
    except ValueError as e:
        return (f"❌ 重命名失败: {e}",) + (gr.update(),) * 5
    _reload_datasets()
    return (f"✅ 已重命名为 '{new}'。",
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
        detail, sel = "点左边某一行查看详情。", ""
    msg = (f"已刷新：共 {len(names)} 个数据。"
           + (f"\n⚠️ {warns}" if warns else ""))
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

HERO_HTML = """
<div id="app-hero">
  <h1>🚀 Unsloth GUI Trainer & Playground</h1>
  <p>模型管理、数据处理、训练、监控、对话，一页完成</p>
  <span class="ver">v0.1.0 · Gradio 6 · Unsloth Core</span>
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
        return "未选择数据集。", {}
    parts = []
    merged_params: dict = {}
    for name in selected:
        cfg = find_by_name(DATASETS, name)
        if cfg is None:
            parts.append(f"❌ 找不到配置: {name}")
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
        tip = f"\n\n💡 已按数据集推荐参数建议: {updates}（可在下方手动微调）"
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
        yield f"❌ 参数格式错误（数字框被清空了？）: {e}", gr.skip()
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
    is_upload = kind == "上传文件"
    is_hf = kind == "HuggingFace"
    is_ex = kind == "已有配置"
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
    if kind == "上传文件":
        if not upload_path:
            return "❌ 请先上传文件。", {}, *_empty, *_locked
        try:
            local_path = persist_upload(upload_path)
        except Exception as e:
            return f"❌ 上传文件保存失败: {e}", {}, *_empty, *_locked
        dkind, existing = "upload", None
    elif kind == "HuggingFace":
        dkind, local_path, existing = "hf", "", None
    else:
        dkind, local_path = "existing", ""
        existing = _find(DATASETS, existing_name)
        if existing is None:
            return f"❌ 找不到配置 '{existing_name}'。", {}, *_empty, *_locked
    try:
        text, state = inspect_text(dkind, hf_id or "", hf_split or "train", local_path, existing)
    except Exception as e:
        return f"❌ 读取失败: {e}", {}, *_empty, *_locked
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
        return f"❌ 预览失败: {e}"


def _prep_generate(state, ins_cols, input_cols, think_cols, out_cols, fixed_ins, name,
                   progress=gr.Progress(track_tqdm=True)):
    try:
        status, preview, new_name = generate_unified(
            state or {}, ins_cols, input_cols, think_cols, out_cols, name,
            fixed_instruction=fixed_ins or "", progress=progress)
    except Exception as e:
        return f"❌ 生成失败: {e}", "", gr.update(), gr.update(), gr.update()
    warns = _reload_datasets()
    msg = status + (f"\n⚠️ 配置告警: {warns}" if warns else "")
    return (
        msg, preview,
        gr.update(choices=DATASET_DISPLAY_NAMES, value=[new_name]),
        gr.update(choices=ALL_DATASET_NAMES),
        gr.update(value=_data_rows()),
    )


def _refresh_train_datasets():
    warns = _reload_datasets()
    msg = (f"已刷新：可用统一数据 {len(DATASET_DISPLAY_NAMES)} 个"
           + (f"\n⚠️ {warns}" if warns else "")
           + ("\n还没有？去「数据处理」Tab 制一份。" if not DATASET_DISPLAY_NAMES else ""))
    return msg, gr.update(choices=DATASET_DISPLAY_NAMES)


with gr.Blocks() as demo:
    gr.HTML(HERO_HTML)
    if CONFIG_WARNINGS:
        gr.Markdown("⚠️ " + "\n\n⚠️ ".join(CONFIG_WARNINGS))
    if is_training():
        gr.Markdown(f"⚠️ 检测到有训练正在运行: {current_experiment()}")

    with gr.Tabs():
        with gr.Tab("🤖 模型管理 (Models)"):
            gr.Markdown("## 🤖 模型管理")
            gr.Markdown(
                "点左边表格某一行，右边直接改；本地路径 / HuggingFace / 魔搭都能加。"
                "改完训练和测试的模型下拉自动刷新。"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    models_table = gr.Dataframe(
                        headers=["展示名", "来源", "模型ID/路径", "4bit", "精度", "上下文长度"],
                        datatype=["str"] * 6, row_count=(0, "dynamic"), column_count=6,
                        interactive=False, wrap=True, value=_model_rows(),
                    )
                    m_refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                with gr.Column(scale=2):
                    m_display = gr.Textbox(label="展示名",
                                           placeholder="如 Qwen3-8B 魔搭版")
                    m_source = gr.Radio(MODEL_SOURCE_OPTIONS, value="HuggingFace",
                                        label="来源")
                    m_id = gr.Textbox(
                        label="模型 ID / 本地路径",
                        placeholder="本地填路径（相对项目根或绝对）；远端填 ID，如 unsloth/Qwen3-8B",
                        lines=2,
                    )
                    with gr.Row():
                        m_4bit = gr.Checkbox(label="4bit 量化加载", value=True)
                        m_dtype = gr.Dropdown(label="精度",
                                              choices=["自动", "bfloat16", "float16"],
                                              value="自动")
                    with gr.Row():
                        m_seq = gr.Number(label="上下文长度", value=2048,
                                          minimum=256, step=256)
                        m_chat = gr.Textbox(label="chat_template（可选）",
                                            placeholder="如 qwen-2.5，留空不强制")
                    with gr.Row():
                        m_new_btn = gr.Button("➕ 新建", size="sm")
                        m_save_btn = gr.Button("💾 保存", variant="primary")
                        m_del_btn = gr.Button("🗑 删除", variant="stop")
                    m_status = gr.Textbox(label="操作状态", interactive=False,
                                          lines=3, max_lines=8)
                    gr.Markdown(
                        "魔搭模型首次使用自动下载；本地路径添加时校验存在性；"
                        "删除要点两次确认。",
                        elem_classes=["hint"],
                    )

        with gr.Tab("🗂 数据管理 (Data)"):
            gr.Markdown("## 🗂 数据管理")
            gr.Markdown(
                "点左边某一行看详情；只删本工具生成的产物（统一数据目录 + 配置），"
                "原始上传文件和自带示例不动。"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    data_table = gr.Dataframe(
                        headers=["名称", "行数", "来源", "创建", "大小", "可训练"],
                        datatype=["str"] * 6, row_count=(0, "dynamic"), column_count=6,
                        interactive=False, wrap=True, value=_data_rows(),
                    )
                    d_refresh_btn = gr.Button("🔄 刷新", size="sm")
                with gr.Column(scale=3):
                    d_detail = gr.Textbox(label="详情", interactive=False,
                                          lines=10, max_lines=20,
                                          value="点左边某一行查看详情。")
                    d_sample = gr.Textbox(label="样本预览（点按钮才加载，远端会下载）",
                                          interactive=False, lines=10, max_lines=20)
                    d_new_name = gr.Textbox(label="新名称（重命名用）",
                                            placeholder="如 wukong_v2")
                    with gr.Row():
                        d_sample_btn = gr.Button("👁 看样本", variant="secondary", size="sm")
                        d_rename_btn = gr.Button("✏️ 重命名", size="sm")
                        d_del_btn = gr.Button("🗑 删除", variant="stop", size="sm")
                    d_status = gr.Textbox(label="操作状态", interactive=False,
                                          lines=3, max_lines=8)
            d_selected = gr.State("")

        with gr.Tab("🧹 数据处理 (Data Prep)"):
            gr.Markdown("## 🧹 把任意数据制成统一训练数据")
            gr.Markdown(
                "**步骤1**: 选来源并「读取列信息」（只取 1 行预览，秒开）。"
                "**步骤2**: 映射哪几列是输入 / 上下文 / 思维链 / 回复（都可多选）。"
                "**步骤3**: 「生成统一训练数据」（这时才全量拉取）。只有这里制成的数据才能拿去训练。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("1. 选择来源", open=True):
                        prep_source_radio = gr.Radio(
                            ["上传文件", "HuggingFace", "已有配置"],
                            value="上传文件", label="数据来源",
                        )
                        prep_upload = gr.File(
                            label="上传文件 (.jsonl / .json / .csv / .parquet / .txt)",
                            file_types=[".jsonl", ".json", ".csv", ".parquet", ".txt"],
                            type="filepath", visible=True,
                        )
                        prep_hf_id = gr.Textbox(
                            label="HF 数据集 ID", value="yahma/alpaca-cleaned", visible=False,
                        )
                        prep_hf_split = gr.Textbox(label="切分 (split)", value="train", visible=False)
                        prep_existing = gr.Dropdown(
                            label="已有配置（可拿旧配置重新制一遍）",
                            choices=ALL_DATASET_NAMES,
                            value=ALL_DATASET_NAMES[0] if ALL_DATASET_NAMES else None,
                            visible=False,
                        )
                        prep_inspect_btn = gr.Button("1. 读取列信息", variant="secondary")
                    with gr.Accordion("2. 列映射与生成", open=True):
                        gr.Markdown(
                            "四个角色都可**多选**，多列按顺序换行拼成一段。"
                            "列名乱没关系，把意思一样的列都勾上就行。"
                            "Response 固定为 think + output 拼接（think 没带 <think> 标签会自动套上）。"
                        )
                        prep_ins_col = gr.Dropdown(
                            label="instruction 输入列（可多选）",
                            choices=[], multiselect=True,
                        )
                        prep_fixed_ins = gr.Textbox(
                            label="固定指令（可选，手写）",
                            placeholder="数据里没有 instruction 列就在这里写一句，所有行共用；同时选了列则做统一前缀拼在前面",
                            lines=3,
                        )
                        prep_input_col = gr.Dropdown(
                            label="input 上下文列（可选，可多选）",
                            choices=[], multiselect=True,
                        )
                        prep_think_col = gr.Dropdown(
                            label="think 思维链列（可选，可多选）",
                            choices=[], multiselect=True,
                        )
                        prep_out_col = gr.Dropdown(
                            label="output 回复列（必填，可多选）",
                            choices=[], multiselect=True,
                        )
                        prep_name = gr.Textbox(
                            label="统一数据名称（必填，将出现在训练列表）",
                            placeholder="如 wukong_v1",
                        )
                        with gr.Row():
                            prep_preview_btn = gr.Button("👁 预览这行训练数据", variant="secondary",
                                                         interactive=False)
                            prep_generate_btn = gr.Button("2. 生成统一训练数据", variant="primary",
                                                          interactive=False)
                with gr.Column(scale=2):
                    prep_inspect_output = gr.Textbox(
                        label="列信息与样本", interactive=False, lines=12, max_lines=25,
                    )
                    prep_status = gr.Textbox(label="生成状态", interactive=False, lines=3, max_lines=8)
                    prep_preview = gr.Textbox(
                        label="统一后预览", interactive=False, lines=12, max_lines=25,
                    )
            prep_state = gr.State({})

        with gr.Tab("训练 (Train)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 训练配置")
                    with gr.Accordion("1. 实验设置", open=True):
                        experiment_name_input = gr.Textbox(label="实验名称 (必填)", value="8gb-vram-test")
                        resume_checkbox = gr.Checkbox(label="从断点继续训练", value=False)
                    with gr.Accordion("2. 模型与数据集", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=MODEL_DISPLAY_NAMES,
                            value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                            label="选择模型",
                        )
                        gr.Markdown(
                            "只显示「🧹 数据处理」制成的统一数据。旧数据去隔壁 Tab 处理一遍再回来。"
                        )
                        dataset_dropdown = gr.Dropdown(
                            choices=DATASET_DISPLAY_NAMES,
                            value=(
                                [DATASET_DISPLAY_NAMES[0]]
                                if DATASET_DISPLAY_NAMES
                                else []
                            ),
                            label="选择数据集 (可多选)",
                            multiselect=True,
                        )
                        with gr.Row():
                            refresh_datasets_btn = gr.Button("🔄 刷新数据集列表", size="sm")
                            preview_btn = gr.Button("👁 预览选中数据集", size="sm")
                        truncate_dataset_checkbox = gr.Checkbox(
                            label="截断数据集用于快速测试",
                            value=True,
                            info="取消勾选以使用完整数据集进行正式训练。",
                        )
                        max_samples_input = gr.Number(
                            value=200, label="截断条数 (勾选截断时生效)", minimum=10, maximum=100000, step=10,
                        )
                    with gr.Accordion("3. LoRA 参数", open=False):
                        lora_r_slider = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank (r)")
                        lora_alpha_slider = gr.Slider(4, 128, value=16, step=4, label="LoRA Alpha")
                    with gr.Accordion("4. 训练核心参数", open=True):
                        training_mode_selector = gr.Radio(
                            ["按步数 (Steps)", "按轮次 (Epochs)"],
                            value="按步数 (Steps)", label="训练模式",
                        )
                        num_epochs_slider = gr.Slider(0.1, 10, value=1, step=0.1,
                                                     label="训练轮数 (Epochs)", visible=False)
                        max_steps_slider = gr.Slider(10, 2000, value=100, step=10,
                                                    label="最大训练步数 (Max Steps)", visible=True)
                        save_steps_input = gr.Number(value=50, label="每 N 步保存一次断点", visible=True)
                        batch_size_slider = gr.Slider(1, 16, value=1, step=1, label="Batch Size")
                        grad_accum_slider = gr.Slider(1, 16, value=8, step=1, label="Gradient Accumulation")
                        learning_rate_slider = gr.Slider(1e-5, 5e-4, value=2e-4, step=1e-5, label="学习率")
                        max_seq_len_slider = gr.Slider(512, 8192, value=2048, step=256, label="Max Seq Length")
                    with gr.Row():
                        start_button = gr.Button("开始训练", variant="primary")
                        stop_button = gr.Button("停止训练", variant="stop")
                with gr.Column(scale=3):
                    gr.Markdown("## TensorBoard 监控面板")
                    tb_status = gr.Textbox(label="TensorBoard 状态", interactive=False)
                    tensorboard_view = gr.HTML("<p>启动后显示 TensorBoard。</p>")
            gr.Markdown("---\n## 数据集预览")
            preview_output = gr.Textbox(label="选中数据集长什么样", interactive=False,
                                        lines=8, max_lines=20)
            gr.Markdown("---\n## 训练日志与状态")
            status_output = gr.Textbox(label="训练进行到哪了", interactive=False,
                                       lines=5, max_lines=20)

        with gr.Tab("测试 (Inference Playground)"):
            gr.Markdown("## 🧠 与你训练的模型对话")
            gr.Markdown(
                "**步骤1**: 选基础模型 + LoRA，点载入。**步骤2**: 写系统提示定人设。"
                "**步骤3**: 直接开聊。换模型先点卸载腾显存。"
            )
            with gr.Row():
                inference_model_selector = gr.Dropdown(
                    label="基础模型 (必须与训练时一致)",
                    choices=MODEL_DISPLAY_NAMES,
                    value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                )
                lora_selector_dropdown = gr.Dropdown(
                    label="LoRA 适配器", choices=list_trained_loras()
                )
            with gr.Row():
                load_model_button = gr.Button("▶ 载入模型", variant="primary")
                unload_model_button = gr.Button("⏏ 卸载腾显存")
                refresh_lora_btn = gr.Button("🔄 刷新 LoRA 列表", size="sm")
            load_status_textbox = gr.Textbox(label="模型加载状态", interactive=False)
            system_prompt_textbox = gr.Textbox(
                label="系统提示 (System Prompt)",
                info="为你的 AI 设定一个身份、规则或基调。",
                lines=3,
                value="你现在是齐天大圣孙悟空，请用孙悟空的身份和风格来回答接下来的所有问题。",
            )
            with gr.Accordion("生成参数", open=False):
                max_tokens_slider = gr.Slider(32, 2048, value=256, step=32, label="Max New Tokens")
                temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                top_k_slider = gr.Slider(1, 100, value=40, step=1, label="Top-k")
            chatbot = gr.Chatbot(label="聊天窗口", height=500)
            with gr.Row():
                chat_input_textbox = gr.Textbox(
                    show_label=False, placeholder="输入你的消息...", scale=4, container=False,
                )
                submit_button = gr.Button("发送", variant="primary", scale=1)
                clear_button = gr.Button("清空", scale=1)

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


def main():
    parser = argparse.ArgumentParser(description="Unsloth GUI Trainer v4.0")
    parser.add_argument("--host", default=os.environ.get("GRADIO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=os.environ.get("GRADIO_SHARE", "") == "1",
                        help="是否生成公网 share 链接（默认关闭，安全）")
    parser.add_argument("--tb-port", type=int, default=int(os.environ.get("TB_PORT", "6006")))
    args = parser.parse_args()

    ok, msg, port = launch_tensorboard(port=args.tb_port)
    print(msg)

    demo.queue(max_size=8).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        inbrowser=False, theme=APP_THEME, css=APP_CSS,
    )


if __name__ == "__main__":
    main()
