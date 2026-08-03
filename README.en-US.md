

# 🚀 Unsloth GUI Trainer: A Professional Interactive Fine-tuning Workspace

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Gradio-based graphical web interface designed to greatly simplify the fine-tuning process of Large Language Models (LLMs) using the [Unsloth](https://github.com/unslothai/unsloth) library. With this tool, you can easily configure, launch, monitor, and manage your model fine-tuning experiments without writing a single line of code.

![Application Screenshot](https://github.com/aifeifei798/unsloth-gui/blob/main/images/1.png)

![Application Screenshot](https://github.com/aifeifei798/unsloth-gui/blob/main/images/2.png)

---

## ✨ Core Features

- **Interactive Graphical Interface**: Built with Gradio, all operations can be completed through a web browser.
- **Real-time Training Monitoring**: Integrated with TensorBoard to visualize key metrics like loss and learning rate in real-time.
- **Configuration-Driven**:
  - **Models**: Dynamically manage and select models via a simple `models.json` file.
  - **Datasets**: Flexibly configure and use multiple datasets via JSON files in the `datasets_config/` directory.
- **Multi-dataset Merging**: Supports selecting and automatically merging multiple datasets in a single training run.
- **Flexible Training Modes**: Freely choose to train by "Epochs" or "Steps".
- **Resume from Checkpoint**: Supports pausing training at any time and seamlessly resuming from the latest checkpoint to ensure experiment safety.
- **In-depth Parameter Tuning**: Provides adjustment options for numerous hyperparameters, including LoRA, batch size, optimizers, and more.
- **Hardware Optimization**: Default configurations are optimized for **8GB VRAM** GPUs (e.g., RTX 3070), ready to use out of the box.

---

## 🔧 Installation & Configuration

### 1. Prerequisites

- **Python**: 3.10 or higher.
- **NVIDIA GPU**: Highly recommended. Requires installation of [NVIDIA CUDA 11.8 or 12.1](https://developer.nvidia.com/cuda-toolkit).
- **VRAM**: At least 8GB recommended, which is the practical minimum for fine-tuning 7B models.
- **Git**: Used to clone this repository.

### 2. Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/aifeifei798/unsloth-gui.git
    cd unsloth-gui
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    All required libraries for the project are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### 3. Project Configuration

Before starting training, you need to configure the models and datasets you wish to use.

#### a) Configure Models

Edit the `models.json` file in the root directory to add Unsloth-supported models you want to use.

**Example `models.json`:**
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

#### b) Configure Datasets

In the `datasets_config/` directory, create a `.json` configuration file for each dataset you want to use.

**Example `datasets_config/alpaca_cleaned.json`:**

**instruction:** Instruction

**input:** Input question

**output:** Answer

Note: Format the data according to your own dataset. There are many formats, so it cannot be strictly unified.

HuggingFace Dataset:
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
Local Dataset:
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

## ▶️ Running the Application

After configuration, run the following command in the project root directory:

```bash
python app.py
```

The application will start TensorBoard in the background and provide a local Gradio URL (e.g., `http://127.0.0.1:7860`) and a public sharing URL. Open either in your browser to get started.

---

## 💡 How to Use

1.  **Experiment Setup**:
    - **Experiment Name**: Specify a unique name for your training task. All outputs and logs will be named accordingly.
    - **Resume from Checkpoint**: Check this if you want to resume from a previously interrupted experiment with the same name.

2.  **Model & Datasets**:
    - Select the model and dataset(s) you defined in the configuration files from the dropdown menus (multiple datasets can be selected).

3.  **Parameter Tuning**:
    - Open the collapsible panel to adjust LoRA and other core training parameters according to your needs and hardware configuration.

4.  **Training Mode**:
    - **By Steps**: Suitable for rapid iteration and large-scale datasets.
    - **By Epochs**: Ensures the model completely learns the entire dataset.

5.  **Start Training**:
    - Click the "Start Training" button.
    - Monitor training progress in real-time in the TensorBoard panel on the right.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
