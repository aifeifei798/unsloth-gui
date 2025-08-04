"""
=====================================================================================
🚀 Unsloth + Gradio + TensorBoard: 一个专业的交互式微调工作台 (终极版) 🚀
=====================================================================================
此版本默认配置已为 8GB VRAM 显卡 (如 RTX 3070) 进行优化，以防止内存溢出。
"""

import gradio as gr
import torch
import os
import subprocess
import time
import json
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from transformers.integrations import TensorBoardCallback
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets

# --- 1. 全局核心配置 ---
LOGS_PARENT_DIR = "logs"
OUTPUTS_PARENT_DIR = "outputs"
TENSORBOARD_PROC = None

# --- 2. 辅助函数：配置加载 (与之前版本相同) ---
def load_config_from_json_files(config_path, default_config, error_msg):
    if isinstance(config_path, Path) and config_path.is_dir():
        json_files = list(config_path.glob("*.json"))
        if not json_files: return [default_config], [default_config.get('display_name', 'Default')]
        configs = [json.load(open(f, 'r', encoding='utf-8')) for f in json_files]
        return configs, [cfg['display_name'] for cfg in configs]
    elif Path(config_path).is_file():
        with open(config_path, 'r', encoding='utf-8') as f: configs = json.load(f)
        return configs, [cfg['display_name'] for cfg in configs]
    else:
        print(f"警告: {error_msg}")
        return [default_config], [default_config.get('display_name', 'Default')]

MODELS_CONFIG, MODEL_DISPLAY_NAMES = load_config_from_json_files(Path("models.json"), {}, "模型配置文件 'models.json' 未找到。")
DATASETS_CONFIG, DATASET_DISPLAY_NAMES = load_config_from_json_files(Path("datasets_config"), {}, "数据集配置目录 'datasets_config' 未找到。")

# --- 3. 辅助函数：启动 TensorBoard (与之前版本相同) ---
def launch_tensorboard():
    global TENSORBOARD_PROC
    if TENSORBOARD_PROC is not None and TENSORBOARD_PROC.poll() is None: return
    os.makedirs(LOGS_PARENT_DIR, exist_ok=True)
    TENSORBOARD_PROC = subprocess.Popen(['tensorboard', '--logdir', LOGS_PARENT_DIR, '--host', '0.0.0.0', '--port', '6006'])
    time.sleep(5)

# --- 4. 辅助函数：准备数据集 (与之前版本相同) ---
def prepare_dataset(dataset_config):
    dataset = load_dataset(dataset_config['dataset_id'], split=dataset_config['split'])
    prompt_template = dataset_config['prompt_template']
    column_mappings = dataset_config['input_columns']
    def formatting_prompts_func(examples):
        texts = []
        zipped_columns = zip(*(examples[col_name] for col_name in column_mappings.values()))
        for values in zipped_columns:
            format_dict = dict(zip(column_mappings.keys(), values))
            texts.append(prompt_template.format(**format_dict))
        return {"text": texts}
    dataset = dataset.map(formatting_prompts_func, batched=True)
    # 为加速演示，仅使用前200个样本。在实际使用中可以注释掉这行。
    return dataset.select(range(200))

# --- 5. 核心功能：模型训练函数 (与之前版本相同) ---
def train_model(
    experiment_name, resume_training,
    training_mode, num_epochs, max_steps, save_steps,
    selected_model_name, selected_dataset_names,
    lora_r, lora_alpha,
    batch_size, grad_accum, optimizer, lr,
    progress=gr.Progress(track_tqdm=True)
):
    if not experiment_name or not experiment_name.strip(): return "错误：实验名称不能为空。"
    experiment_name = experiment_name.strip().replace(" ", "_")
    output_dir = Path(OUTPUTS_PARENT_DIR) / experiment_name
    logging_dir = Path(LOGS_PARENT_DIR) / experiment_name
    progress(0, desc=f"准备实验: {experiment_name}")

    model_config = next((item for item in MODELS_CONFIG if item["display_name"] == selected_model_name), None)
    if not model_config or not selected_dataset_names: return "错误：必须选择一个模型和至少一个数据集。"
    progress(0.1, desc="加载并处理数据集...")
    all_datasets = [prepare_dataset(cfg) for name in selected_dataset_names if (cfg := next((item for item in DATASETS_CONFIG if item["display_name"] == name), None))]
    if not all_datasets: return "错误：无法加载所选的数据集配置。"
    progress(0.2, desc=f"合并 {len(all_datasets)} 个数据集中...")
    combined_dataset = concatenate_datasets(all_datasets)

    progress(0.3, desc=f"初始化模型: {model_config['model_id']}...")
    dtype_str = model_config.get('dtype')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(dtype_str)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['model_id'], max_seq_length=2048, dtype=dtype, load_in_4bit=model_config['load_in_4bit'],
    )
    model = FastLanguageModel.get_peft_model(
        model, r=int(lora_r), lora_alpha=int(lora_alpha),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", use_gradient_checkpointing="unslued", random_state=3407,
    )

    progress(0.4, desc="配置训练参数...")
    args = {
        "output_dir": str(output_dir), "logging_dir": str(logging_dir),
        "per_device_train_batch_size": int(batch_size), "gradient_accumulation_steps": int(grad_accum),
        "learning_rate": float(lr), "logging_steps": 1, "optim": optimizer,
        "fp16": not torch.cuda.is_bf16_supported(), "bf16": torch.cuda.is_bf16_supported(),
        "warmup_steps": 10, "weight_decay": 0.01, "lr_scheduler_type": "linear",
        "seed": 3407, "save_total_limit": 3,
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
        model=model, tokenizer=tokenizer, train_dataset=combined_dataset,
        dataset_text_field="text", max_seq_length=2048, dataset_num_proc=2,
        packing=False, args=training_args, callbacks=[TensorBoardCallback()],
    )

    status_msg = "从断点继续训练..." if resume_training else "开始新的训练..."
    progress(0.5, desc=status_msg)
    trainer.train(resume_from_checkpoint=resume_training)
    
    progress(1.0, desc="训练完成！")
    return f"训练完成！模型已保存在 '{output_dir}'"

# --- 6. Gradio UI 界面定义 (已更新默认值) ---
def update_training_mode_ui(mode):
    is_epoch_mode = mode == "按轮次 (Epochs)"
    return {
        num_epochs_slider: gr.update(visible=is_epoch_mode),
        max_steps_slider: gr.update(visible=not is_epoch_mode),
        save_steps_input: gr.update(visible=not is_epoch_mode),
    }

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🚀 Unsloth + Gradio + TensorBoard (终极版)")
    gr.Markdown("### ✨ *默认参数已为 8GB VRAM (如 RTX 3070) 优化*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ 训练配置")
            with gr.Accordion("1. 实验设置", open=True):
                experiment_name_input = gr.Textbox(label="实验名称 (必填)", value="3070-8gb-default-test")
                resume_checkbox = gr.Checkbox(label="从断点继续训练", value=False)
            
            with gr.Accordion("2. 模型与数据集", open=True):
                model_dropdown = gr.Dropdown(choices=MODEL_DISPLAY_NAMES, value=MODEL_DISPLAY_NAMES[0] if MODEL_DISPLAY_NAMES else None, label="选择模型")
                dataset_dropdown = gr.Dropdown(choices=DATASET_DISPLAY_NAMES, value=[DATASET_DISPLAY_NAMES[0]] if DATASET_DISPLAY_NAMES else None, label="选择数据集 (可多选)", multiselect=True)

            with gr.Accordion("3. LoRA 参数 (8GB 优化)", open=False):
                lora_r_slider = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank (r)") # <<< 8GB 优化
                lora_alpha_slider = gr.Slider(4, 128, value=16, step=4, label="LoRA Alpha") # <<< 8GB 优化

            with gr.Accordion("4. 训练核心参数 (8GB 优化)", open=True):
                training_mode_selector = gr.Radio(["按步数 (Steps)", "按轮次 (Epochs)"], value="按步数 (Steps)", label="训练模式")
                num_epochs_slider = gr.Slider(0.1, 10, 3, step=0.1, label="训练轮数 (Epochs)", visible=False)
                max_steps_slider = gr.Slider(10, 2000, 100, step=10, label="最大训练步数 (Max Steps)", visible=True)
                save_steps_input = gr.Number(value=50, label="每 N 步保存一次断点", visible=True)
                
                batch_size_slider = gr.Slider(1, 16, value=1, step=1, label="Batch Size") # <<< 8GB 优化
                grad_accum_slider = gr.Slider(1, 16, value=8, step=1, label="Gradient Accumulation") # <<< 8GB 优化
                optimizer_dropdown = gr.Dropdown(["adamw_8bit", "adamw_torch"], value="adamw_8bit", label="优化器")
                learning_rate_slider = gr.Slider(1e-5, 5e-4, 2e-4, step=1e-5, label="学习率")

            start_button = gr.Button("✅ 开始训练", variant="primary")
            status_output = gr.Textbox(label="训练状态", interactive=False, lines=2)

        with gr.Column(scale=3):
            gr.Markdown("## 📊 TensorBoard 监控面板")
            tensorboard_html = f'<iframe src="http://127.0.0.1:6006" width="100%" height="800px" frameborder="0"></iframe>'
            tensorboard_view = gr.HTML(tensorboard_html)

    training_mode_selector.change(fn=update_training_mode_ui, inputs=training_mode_selector, outputs=[num_epochs_slider, max_steps_slider, save_steps_input])
    
    all_inputs = [
        experiment_name_input, resume_checkbox,
        training_mode_selector, num_epochs_slider, max_steps_slider, save_steps_input,
        model_dropdown, dataset_dropdown, lora_r_slider, lora_alpha_slider,
        batch_size_slider, grad_accum_slider, optimizer_dropdown, learning_rate_slider
    ]
    start_button.click(fn=train_model, inputs=all_inputs, outputs=[status_output])
    
    demo.load(fn=launch_tensorboard)

# --- 7. 启动应用 ---
if __name__ == "__main__":
    demo.launch(share=True)