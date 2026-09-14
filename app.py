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
DATASET_DISPLAY_NAMES = [d.display_name for d in DATASETS]


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
    req = TrainRequest(
        experiment_name=experiment_name, resume_training=bool(resume_training),
        truncate_dataset=bool(truncate_dataset), max_samples=int(max_samples),
        training_mode=training_mode, num_epochs=float(num_epochs),
        max_steps=int(max_steps), save_steps=int(save_steps),
        selected_model_name=selected_model_name,
        selected_dataset_names=list(selected_dataset_names or []),
        lora_r=int(lora_r), lora_alpha=int(lora_alpha),
        batch_size=int(batch_size), grad_accum=int(grad_accum),
        lr=float(lr), max_seq_length=int(max_seq_length),
    )
    last = ""
    for msg in run_training(req, progress=progress):
        last = msg
        yield last
    # 训练结束刷新 LoRA 下拉
    try:
        yield last
    except Exception:
        pass


def _stop_training():
    return request_cancel()


def _load_model_wrapper(base, lora, progress=gr.Progress(track_tqdm=True)):
    msg = load_inference_model(base, lora, progress=progress)
    return f"{msg}\n{loaded_info()}", gr.update(choices=list_trained_loras())


with gr.Blocks() as demo:
    gr.Markdown("# Unsloth GUI Trainer & Playground (v4.0)")
    if CONFIG_WARNINGS:
        gr.Markdown("⚠️ " + "\n\n⚠️ ".join(CONFIG_WARNINGS))
    if is_training():
        gr.Markdown(f"⚠️ 检测到有训练正在运行: {current_experiment()}")

    with gr.Tabs():
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
                        dataset_dropdown = gr.Dropdown(
                            choices=DATASET_DISPLAY_NAMES,
                            value=[DATASET_DISPLAY_NAMES[0]] if DATASET_DISPLAY_NAMES else [],
                            label="选择数据集 (可多选)",
                            multiselect=True,
                        )
                        truncate_dataset_checkbox = gr.Checkbox(
                            label="截断数据集用于快速测试",
                            value=True,
                            info="取消勾选以使用完整数据集进行正式训练。",
                        )
                        max_samples_input = gr.Number(
                            value=200, label="截断条数 (勾选截断时生效)", minimum=10, maximum=100000, step=10,
                        )
                        preview_btn = gr.Button("预览数据集格式")
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
            preview_output = gr.Textbox(label="Preview", interactive=False, lines=8, max_lines=20)
            gr.Markdown("---\n## 训练日志与状态")
            status_output = gr.Textbox(label="Status", interactive=False, lines=5, max_lines=20)

        with gr.Tab("测试 (Inference Playground)"):
            gr.Markdown("## 与你训练的模型对话")
            gr.Markdown("**步骤1**: 选择基础模型。**步骤2**: 选择 LoRA 适配器。**步骤3**: 设定系统提示。")
            with gr.Row():
                inference_model_selector = gr.Dropdown(
                    label="选择基础模型 (必须与训练时一致)",
                    choices=MODEL_DISPLAY_NAMES,
                    value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                )
                lora_selector_dropdown = gr.Dropdown(
                    label="选择 LoRA 适配器", choices=list_trained_loras()
                )
                refresh_lora_btn = gr.Button("刷新 LoRA 列表")
            with gr.Row():
                load_model_button = gr.Button("载入模型进行测试", variant="primary")
                unload_model_button = gr.Button("卸载模型释放显存")
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
    start_button.click(fn=_train_wrapper, inputs=train_inputs, outputs=[status_output])
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
        inbrowser=False, theme=gr.themes.Soft(), css="footer {display: none !important}",
    )


if __name__ == "__main__":
    main()
