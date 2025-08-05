"""
=====================================================================================
🚀 Unsloth GUI Trainer & Playground (v3.8 - 数据截断控制) 🚀
=====================================================================================
此版本根据用户反馈，新增了核心的易用性功能：
1.  在UI界面上增加了一个复选框，允许用户自由选择是否截断数据集用于快速测试。
2.  默认开启截断，以保证新手的初次体验流畅、快速。
3.  移除了旧的、基于JSON配置的 use_full_dataset 标志，将控制权完全交由UI。
"""

import gradio as gr
import torch
import os
import subprocess
import time
import json
from pathlib import Path
from threading import Thread

# 提高 PyTorch Dynamo 的重编译容忍上限
torch._dynamo.config.recompile_limit = 100

from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextIteratorStreamer
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, load_from_disk

# --- (前面的代码，如全局配置和辅助函数，保持不变) ---
# --- 1. 全局核心配置 ---
LOGS_PARENT_DIR = "logs"
OUTPUTS_PARENT_DIR = "outputs"
TENSORBOARD_PROC = None


# --- 2. 辅助函数：配置加载 ---
def load_config_from_json_files(config_path, default_config, error_msg):
    if isinstance(config_path, Path) and config_path.is_dir():
        json_files = list(config_path.glob("*.json"))
        if not json_files:
            return [default_config], [default_config.get("display_name", "Default")]
        configs = [json.load(open(f, "r", encoding="utf-8")) for f in json_files]
        return configs, [cfg["display_name"] for cfg in configs]
    elif Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        return configs, [cfg["display_name"] for cfg in configs]
    else:
        print(f"警告: {error_msg}")
        return [default_config], [default_config.get("display_name", "Default")]


MODELS_CONFIG, MODEL_DISPLAY_NAMES = load_config_from_json_files(
    Path("models.json"), {}, "模型配置文件 'models.json' 未找到。"
)
DATASETS_CONFIG, DATASET_DISPLAY_NAMES = load_config_from_json_files(
    Path("datasets_config"), {}, "数据集配置目录 'datasets_config' 未找到。"
)


# --- 3. 辅助函数：启动 TensorBoard ---
def launch_tensorboard():
    global TENSORBOARD_PROC
    if TENSORBOARD_PROC is not None and TENSORBOARD_PROC.poll() is None:
        return
    os.makedirs(LOGS_PARENT_DIR, exist_ok=True)
    TENSORBOARD_PROC = subprocess.Popen(
        [
            "tensorboard",
            "--logdir",
            LOGS_PARENT_DIR,
            "--host",
            "0.0.0.0",
            "--port",
            "6006",
        ]
    )
    time.sleep(5)


# --- 4. 辅助函数：准备数据集 (已更新) ---
def prepare_dataset(dataset_config, truncate_for_testing):
    """根据配置加载数据，并根据UI选项决定是否截断。"""
    if dataset_config.get("is_local", False):
        dataset_path = dataset_config["dataset_id"]
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"本地数据集路径未找到: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(
            dataset_config["dataset_id"], split=dataset_config.get("split", "train")
        )

    prompt_template = dataset_config["prompt_template"]
    column_mappings = dataset_config["input_columns"]

    def formatting_prompts_func(examples):
        texts = []
        zipped_columns = zip(
            *(examples[col_name] for col_name in column_mappings.values())
        )
        for values in zipped_columns:
            format_dict = dict(zip(column_mappings.keys(), values))
            texts.append(prompt_template.format(**format_dict))
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # vvvvvvvvvvvvvv 这是修改的地方 vvvvvvvvvvvvvv
    # 根据 UI 复选框的值来决定是否截断
    if truncate_for_testing and len(dataset) > 200:
        print("截断数据集至前200条用于快速测试。")
        dataset = dataset.select(range(200))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return dataset


# --- 5. 核心功能：模型训练函数 (已更新) ---
def train_model(
    # -- 新增 truncate_dataset 参数 --
    experiment_name,
    resume_training,
    truncate_dataset,
    # -- 其他参数 --
    training_mode,
    num_epochs,
    max_steps,
    save_steps,
    selected_model_name,
    selected_dataset_names,
    lora_r,
    lora_alpha,
    batch_size,
    grad_accum,
    lr,
    progress=gr.Progress(track_tqdm=True),
):
    # ... (前面的代码不变) ...
    if not experiment_name or not experiment_name.strip():
        return "错误：实验名称不能为空。"
    experiment_name = experiment_name.strip().replace(" ", "_")
    output_dir = Path(OUTPUTS_PARENT_DIR) / experiment_name
    logging_dir = Path(LOGS_PARENT_DIR) / experiment_name
    progress(0, desc=f"准备实验: {experiment_name}")

    model_config = next(
        (item for item in MODELS_CONFIG if item["display_name"] == selected_model_name),
        None,
    )
    if not model_config or not selected_dataset_names:
        return "错误：必须选择一个模型和至少一个数据集。"
    progress(0.1, desc="加载并处理数据集...")

    # vvvvvvvvvvvvvv 这是修改的地方 vvvvvvvvvvvvvv
    # 将 UI 复选框的值传递给 prepare_dataset 函数
    all_datasets = [
        prepare_dataset(cfg, truncate_dataset)
        for name in selected_dataset_names
        if (
            cfg := next(
                (item for item in DATASETS_CONFIG if item["display_name"] == name), None
            )
        )
    ]
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if not all_datasets:
        return "错误：无法加载所选的数据集配置。"
    progress(0.2, desc=f"合并 {len(all_datasets)} 个数据集中...")
    combined_dataset = concatenate_datasets(all_datasets)

    # ... (后续训练逻辑与上一版完全相同) ...
    progress(0.3, desc=f"初始化模型: {model_config['model_id']}...")
    dtype_str = model_config.get("dtype")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["model_id"],
        max_seq_length=2048,
        dtype=dtype,
        load_in_4bit=model_config["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    progress(0.4, desc="配置训练参数...")
    args = {
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "learning_rate": float(lr),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "save_total_limit": 3,
    }
    if training_mode == "按轮次 (Epochs)":
        args["num_train_epochs"] = float(num_epochs)
        args["save_strategy"] = "epoch"
    else:
        args["max_steps"] = int(max_steps)
        args["save_strategy"] = "steps"
        args["save_steps"] = int(save_steps)
    training_args = TrainingArguments(**args)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[TensorBoardCallback()],
    )
    status_msg = "从断点继续训练..." if resume_training else "开始新的训练..."
    progress(0.5, desc=status_msg)
    trainer.train(resume_from_checkpoint=resume_training)
    progress(0.9, desc="训练完成，正在保存最终 LoRA 适配器...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    progress(1.0, desc="所有任务完成！")
    return f"训练完成！最终 LoRA 适配器已保存在 '{output_dir}'"


# --- 6. 推理 (Inference) 相关函数 (保持不变) ---
# ... (load_inference_model, run_chat 等函数代码不变) ...
def list_trained_loras():
    if not Path(OUTPUTS_PARENT_DIR).exists():
        return []
    return [d.name for d in Path(OUTPUTS_PARENT_DIR).iterdir() if d.is_dir()]


def load_inference_model(
    base_model_name, lora_name, progress=gr.Progress(track_tqdm=True)
):
    progress(0, desc="开始加载模型...")
    if not base_model_name or not lora_name:
        return None, None, "错误：必须同时选择一个基础模型和一个 LoRA 适配器。"
    model_config = next(
        (item for item in MODELS_CONFIG if item["display_name"] == base_model_name),
        None,
    )
    if not model_config:
        return None, None, f"错误：找不到基础模型 '{base_model_name}' 的配置。"
    base_model_id = model_config["model_id"]
    lora_path = Path(OUTPUTS_PARENT_DIR) / lora_name
    if not lora_path.exists():
        return None, None, f"错误：LoRA 目录未找到: {lora_path}"
    try:
        progress(0.2, desc=f"正在加载基础模型: {base_model_id}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id, max_seq_length=2048, load_in_4bit=True
        )
        progress(0.6, desc=f"正在应用 LoRA 适配器: {lora_name}...")
        model.load_adapter(str(lora_path))
        progress(1.0, desc="模型加载成功！")
        return model, tokenizer, f"✅ 模型 '{lora_name}' 加载成功！可以开始对话了。"
    except Exception as e:
        return None, None, f"❌ 模型加载失败: {e}"


def run_chat(user_input, history, system_prompt, model, tokenizer):
    history = history or []
    if model is None or tokenizer is None:
        history.append({"role": "user", "content": user_input})
        history.append(
            {
                "role": "assistant",
                "content": "错误：模型未加载。请先选择并加载一个模型。",
            }
        )
        return history
    messages = (
        [{"role": "system", "content": system_prompt}]
        if system_prompt and system_prompt.strip()
        else []
    )
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    prompt_inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=prompt_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        use_cache=True,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    history.append({"role": "assistant", "content": ""})
    for new_text in streamer:
        history[-1]["content"] += new_text
        yield history


# --- 7. Gradio UI 界面定义 (已更新) ---
def update_training_mode_ui(mode):
    # ... (代码不变) ...
    is_epoch_mode = mode == "按轮次 (Epochs)"
    return {
        num_epochs_slider: gr.update(visible=is_epoch_mode),
        max_steps_slider: gr.update(visible=not is_epoch_mode),
        save_steps_input: gr.update(visible=not is_epoch_mode),
    }


with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🚀 Unsloth GUI Trainer & Playground (v3.8)")

    loaded_model_state = gr.State(None)
    loaded_tokenizer_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("🚀 训练 (Train)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ 训练配置")
                    with gr.Accordion("1. 实验设置", open=True):
                        experiment_name_input = gr.Textbox(
                            label="实验名称 (必填)", value="8gb-vram-test"
                        )
                        resume_checkbox = gr.Checkbox(
                            label="从断点继续训练", value=False
                        )
                    with gr.Accordion("2. 模型与数据集", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=MODEL_DISPLAY_NAMES,
                            value=(
                                MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None
                            ),
                            label="选择模型",
                        )
                        dataset_dropdown = gr.Dropdown(
                            choices=DATASET_DISPLAY_NAMES,
                            value=(
                                [DATASET_DISPLAY_NAMES[0]]
                                if DATASET_DISPLAY_NAMES
                                else None
                            ),
                            label="选择数据集 (可多选)",
                            multiselect=True,
                        )
                        # vvvvvvvvvvvvvv 这是新增的地方 vvvvvvvvvvvvvv
                        truncate_dataset_checkbox = gr.Checkbox(
                            label="截断数据集用于快速测试 (仅使用前200条)",
                            value=True,
                            info="取消勾选以使用完整数据集进行正式训练。",
                        )
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    with gr.Accordion("3. LoRA 参数 (8GB 优化)", open=False):
                        lora_r_slider = gr.Slider(
                            4, 64, value=8, step=4, label="LoRA Rank (r)"
                        )
                        lora_alpha_slider = gr.Slider(
                            4, 128, value=16, step=4, label="LoRA Alpha"
                        )
                    with gr.Accordion("4. 训练核心参数 (8GB 优化)", open=True):
                        training_mode_selector = gr.Radio(
                            ["按步数 (Steps)", "按轮次 (Epochs)"],
                            value="按步数 (Steps)",
                            label="训练模式",
                        )
                        num_epochs_slider = gr.Slider(
                            0.1,
                            10,
                            value=1,
                            step=0.1,
                            label="训练轮数 (Epochs)",
                            visible=False,
                        )
                        max_steps_slider = gr.Slider(
                            10,
                            2000,
                            100,
                            step=10,
                            label="最大训练步数 (Max Steps)",
                            visible=True,
                        )
                        save_steps_input = gr.Number(
                            value=50, label="每 N 步保存一次断点", visible=True
                        )
                        batch_size_slider = gr.Slider(
                            1, 16, value=1, step=1, label="Batch Size"
                        )
                        grad_accum_slider = gr.Slider(
                            1, 16, value=8, step=1, label="Gradient Accumulation"
                        )
                        learning_rate_slider = gr.Slider(
                            1e-5, 5e-4, 2e-4, step=1e-5, label="学习率"
                        )
                    start_button = gr.Button("✅ 开始训练", variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("## 📊 TensorBoard 监控面板")
                    tensorboard_html = f'<iframe src="http://127.0.0.1:6006" width="100%" height="1188px" frameborder="0"></iframe>'
                    tensorboard_view = gr.HTML(tensorboard_html)
            gr.Markdown("--- \n ## 📋 训练日志与状态")
            status_output = gr.Textbox(
                label="Status", interactive=False, lines=3, max_lines=5
            )

        with gr.Tab("💬 测试 (Inference Playground)"):
            # ... (测试 Tab 的代码不变) ...
            gr.Markdown("## 🧠 与你训练的模型对话")
            gr.Markdown(
                "**步骤1**: 选择基础模型。**步骤2**: 选择 LoRA 适配器。**步骤3**: 设定系统提示。"
            )
            with gr.Row():
                inference_model_selector = gr.Dropdown(
                    label="选择基础模型 (必须与训练时一致)",
                    choices=MODEL_DISPLAY_NAMES,
                    value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None,
                )
                lora_selector_dropdown = gr.Dropdown(
                    label="选择 LoRA 适配器", choices=list_trained_loras()
                )
            load_model_button = gr.Button("载入模型进行测试", variant="primary")
            load_status_textbox = gr.Textbox(label="模型加载状态", interactive=False)
            system_prompt_textbox = gr.Textbox(
                label="系统提示 (System Prompt)",
                info="为你的 AI 设定一个身份、规则或基调。",
                lines=3,
                value="你现在是齐天大圣孙悟空，请用孙悟空的身份和风格来回答接下来的所有问题。",
            )
            chatbot = gr.Chatbot(label="聊天窗口", type="messages", height=500)
            with gr.Row():
                chat_input_textbox = gr.Textbox(
                    show_label=False,
                    placeholder="输入你的消息...",
                    scale=4,
                    container=False,
                )
                submit_button = gr.Button("发送", variant="primary", scale=1)

    training_mode_selector.change(
        fn=update_training_mode_ui,
        inputs=training_mode_selector,
        outputs=[num_epochs_slider, max_steps_slider, save_steps_input],
    )

    train_inputs = [
        experiment_name_input,
        resume_checkbox,
        truncate_dataset_checkbox,  # <-- 已添加
        training_mode_selector,
        num_epochs_slider,
        max_steps_slider,
        save_steps_input,
        model_dropdown,
        dataset_dropdown,
        lora_r_slider,
        lora_alpha_slider,
        batch_size_slider,
        grad_accum_slider,
        learning_rate_slider,
    ]
    start_button.click(fn=train_model, inputs=train_inputs, outputs=[status_output])

    load_model_button.click(
        fn=load_inference_model,
        inputs=[inference_model_selector, lora_selector_dropdown],
        outputs=[loaded_model_state, loaded_tokenizer_state, load_status_textbox],
    )
    submit_event = chat_input_textbox.submit(
        fn=run_chat,
        inputs=[
            chat_input_textbox,
            chatbot,
            system_prompt_textbox,
            loaded_model_state,
            loaded_tokenizer_state,
        ],
        outputs=[chatbot],
    )
    submit_event.then(lambda: gr.update(value=""), outputs=[chat_input_textbox])
    button_event = submit_button.click(
        fn=run_chat,
        inputs=[
            chat_input_textbox,
            chatbot,
            system_prompt_textbox,
            loaded_model_state,
            loaded_tokenizer_state,
        ],
        outputs=[chatbot],
    )
    button_event.then(lambda: gr.update(value=""), outputs=[chat_input_textbox])

    demo.load(fn=launch_tensorboard)

# --- 8. 启动应用 ---
if __name__ == "__main__":
    demo.launch(share=True)
