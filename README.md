# 🚀 Unsloth GUI Trainer: 一个专业的交互式微调工作台

> v4.0（轻量专属路线，不跟 Unsloth Studio 比大而全）：单文件拆为 `src/` 模块、
> 训练可取消、TRL 新旧 API 兼容、数据集校验+预览、推理显存管理、TB 端口探测。
> 旧 `python app.py` 依然可用，新启动参数见下文。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于 Gradio 的图形化 Web 界面，旨在极大简化使用 [Unsloth](https://github.com/unslothai/unsloth) 库进行大语言模型（LLM）的微调过程。通过本工具，您可以轻松配置、启动、监控和管理您的模型微调实验，无需编写一行代码。

![Application Screenshot](https://github.com/aifeifei798/unsloth-gui/blob/main/images/1.png)

![Application Screenshot](https://github.com/aifeifei798/unsloth-gui/blob/main/images/2.png)

---

## ✨ 核心功能

- **交互式图形界面**: 基于 Gradio 构建，所有操作皆可通过浏览器完成。
- **实时训练监控**: 集成 TensorBoard，实时可视化损失、学习率等关键指标。
- **配置驱动**:
  - **模型**: 通过简单的 `models.json` 文件动态管理和选择模型。
  - **数据集**: 通过 `datasets_config/` 目录下的 JSON 文件灵活配置和使用多个数据集。
- **多数据集合并**: 支持在一次训练中选择并自动合并多个数据集。
- **灵活的训练模式**: 可自由选择按“轮次 (Epochs)”或“步数 (Steps)”进行训练。
- **断点续训**: 支持随时中断训练，并能从最新的断点无缝恢复，确保实验安全。
- **深度参数调优**: 开放了 LoRA、批次大小、优化器等大量超参数的调节选项。
- **硬件优化**: 默认配置已为 **8GB VRAM** 显卡（如 RTX 3070）进行优化，开箱即用。

---

## 🔧 安装与配置

### 1. 先决条件

- **Python**: 3.10 或更高版本。
- **NVIDIA GPU**: 强烈推荐。需要安装 [NVIDIA CUDA 11.8 或 12.1](https://developer.nvidia.com/cuda-toolkit)。
- **显存 (VRAM)**: 建议至少 8GB，这是微调 7B 模型的实际最低要求。
- **Git**: 用于克隆本仓库。

### 2. 安装步骤

1.  **克隆仓库**
    ```bash
    git clone https://github.com/aifeifei798/unsloth-gui.git
    cd unsloth-gui
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # 在 Windows 上，使用: venv\Scripts\activate
    ```

3.  **安装依赖**（unsloth 会自动带上 torch 及训练栈，不用手动装）
    ```bash
    pip install -r requirements.txt
    # 或用 uv 自动匹配 torch 后端（推荐）：
    # uv pip install -r requirements.txt --torch-backend=auto
    ```

### 3. 项目配置

在开始训练之前，您需要配置想要使用的模型和数据集。

#### a) 配置模型

编辑根目录下的 `models.json` 文件，添加您想使用的 Unsloth 支持的模型。

**示例 `models.json`:**
```json
[
  {
    "display_name": "gemma-3-1b-it-qat-q4_0-unquantized",
    "model_id": "../gemma-3-1b-it-qat-q4_0-unquantized", //本地模型
    "load_in_4bit": true,
    "dtype": null
  },
  {
    "display_name": "Mistral 7B Instruct v0.2 (4-bit)",
    "model_id": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "load_in_4bit": true,
    "dtype": null
  },
  {
    "display_name": "Llama-3 8B Instruct (4-bit)",
    "model_id": "unsloth/llama-3-8b-instruct-bnb-4bit",
    "load_in_4bit": true,
    "dtype": null
  },
  {
    "display_name": "Phi-3 Mini 4k Instruct (bf16)",
    "model_id": "unsloth/phi-3-mini-4k-instruct",
    "load_in_4bit": false,
    "dtype": "bfloat16"
  }
]
```

#### b) 准备数据集（必须走「🧹 数据处理」Tab）

训练只认处理过的统一数据。流程：打开「🧹 数据处理」Tab →
选来源（上传文件 / HuggingFace ID / 已有配置）→「读取列信息」→
映射 `instruction` 输入列、`input` 上下文列（可选）、`think` 思维链列（可选）、
`output` 回复列 →「生成统一训练数据」。生成后去「训练」Tab 点「🔄 刷新数据集列表」即可选中。

统一格式为 `instruction / input / think / output` 四列，每个角色都可多选，
多列按顺序换行拼成一段；没映射 input 就没有对应段落；
最终训练文本恒为 Instruction / Input(可选) / Response，
其中 Response = think + output 拼接；空回复的行会自动丢弃并计数。产物在 `local_data/processed/<名称>/`
（`data.jsonl` + `hf_dataset/` + `manifest.json`），配置自动写入
`datasets_config/`，不用手写 JSON。

**示例 `datasets_config/alpaca_cleaned.json`:**

**instruction:** 系统词

**input:** 输入问题

**output:** 答案

说明: 根据数据自己来整合出数据,这个格式很多,无法统一

hf数据
```json
{
  "display_name": "Alpaca (Cleaned)",
  "dataset_id": "yahma/alpaca-cleaned",
  "split": "train",
  "prompt_template": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
  "input_columns": {
    "instruction": "instruction",
    "input": "input",
    "output": "output"
  }
}
```
本地数据
```json
{
  "display_name": "Chinese-DeepSeek-R1-Distill-data-110k-alpaca",
  "dataset_id": "../Chinese-DeepSeek-R1-Distill-data-110k-alpaca",
  "split": "train",
  "is_local": true, //说明是本地数据
  "prompt_template": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
  "input_columns": {
    "instruction": "instruction",
    "input": "input",
    "output": "output"
  }
}
```

---

## ▶️ 运行应用

配置完成后，在项目根目录运行以下命令：

```bash
python app.py
# 可选参数：--host 127.0.0.1 --port 7860 --tb-port 6006 --share
# 默认不开启公网 share（安全），需要时加 --share 或 GRADIO_SHARE=1
python app.py --host 0.0.0.0 --port 7860
```

应用将在后台启动 TensorBoard（端口被占会自动顺延），并提供本地 Gradio 网址
(如 `http://127.0.0.1:7860`)。新版训练支持“停止训练”安全中断、
数据集预览、截断条数可调、推理参数可调、LoRA 列表刷新与模型卸载。

---

## 💡 如何使用

1.  **实验设置**:
    - **实验名称**: 为您的训练任务指定一个唯一的名称。所有输出和日志都将以此命名。
    - **从断点继续训练**: 如果您想从之前中断的同名实验中恢复，请勾选此项。

2.  **模型与数据集**:
    - 从下拉框中选择您在配置文件中定义的模型和数据集（可多选）。

3.  **参数调整**:
    - 打开折叠面板，根据您的需求和硬件配置，调整 LoRA 和其他核心训练参数。

4.  **训练模式**:
    - **按步数 (Steps)**: 适合快速迭代和大规模数据集。
    - **按轮次 (Epochs)**: 确保模型完整地学习整个数据集。

5.  **开始训练**:
    - 点击“开始训练”按钮。
    - 在右侧的 TensorBoard 面板中实时监控训练进度。

---

## 📄 License

本项目采用 [MIT License](LICENSE) 授权。
