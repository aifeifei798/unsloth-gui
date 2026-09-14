# 🚀 Unsloth GUI Trainer：轻量专属微调工作台

[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [English](README.en.md)

一个基于 Gradio + [Unsloth](https://github.com/unslothai/unsloth) 的图形化微调工具，
专注**纯文本单卡 SFT**：模型管理、数据处理、训练、监控、对话，一个页面完成，
全程不用写代码，不用手改配置文件。

> 只做文本训练（带图/视频/音频暂不支持）。8GB 显存开箱即用。

---

## ✨ 五个 Tab

| Tab | 干什么 |
| --- | ------ |
| 🤖 模型管理 | 点行即改：本地路径 / HuggingFace / 魔搭 ModelScope 三种来源增删改，同名覆盖更新，删除要点两次确认 |
| 🗂 数据管理 | 统一数据的列表、行数、大小、映射详情，看样本、重命名（产物跟着搬家）、删除（只删生成物，源文件不动） |
| 🧹 数据处理 | 上传文件 / HF ID / 已有配置 → 只取 1 行秒看列 → 四角色多选映射（instruction / input / think / output）→ 单行预览 → 生成统一训练数据 |
| 🚀 训练 | 按步数/轮次、可取消、多数据集合并、断点续训、TensorBoard 实时看板 |
| 💬 测试 | 载入 LoRA 流式对话，温度/top-p 等可调，一键卸载腾显存 |

统一数据格式：`instruction / input / think / output` 四列，多列换行拼接；
最终训练文本恒为 Instruction / Input（可选）/ Response，其中
Response = think + output（think 缺 `<think>` 标签自动套上）；
空回复行自动丢弃。只有这里产出的数据才能拿去训练。

---

## 🔧 安装

先决条件：Python 3.10+，NVIDIA GPU + CUDA（8GB 显存起步）。

```bash
git clone https://github.com/aifeifei798/unsloth-gui.git
cd unsloth-gui

python3 -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

pip install -r requirements.txt
# 或用 uv 自动匹配 torch 后端（推荐）：
# uv pip install -r requirements.txt --torch-backend=auto
```

依赖只有 5 个直接项：`unsloth`、`bitsandbytes`、`gradio`、`tensorboard`、
`hf_transfer`、`modelscope`，训练栈（torch / transformers / trl 等）跟着 unsloth 走。

---

## ▶️ 运行

```bash
python app.py
# 可选参数：--host 127.0.0.1 --port 7860 --tb-port 6006 --share
# 默认不开启公网 share，需要时加 --share 或 GRADIO_SHARE=1
```

浏览器打开 `http://127.0.0.1:7860`，流程：模型管理加模型 →
数据处理制数据 → 训练（右侧 TensorBoard 实时监控，支持中途停止）→
测试 Tab 载入 LoRA 对话。

产物：`outputs/<实验名>/`（LoRA 适配器）、`logs/<实验名>/`（TensorBoard）、
`local_data/processed/<数据名>/`（统一数据）。

---

## 📁 目录结构

```
app.py                  # 薄 UI 层（Gradio）
src/
  config.py             # 模型/数据集配置加载、校验、来源解析
  dataset_utils.py      # 训练格式化、空段清理、多数据集合并
  dataprep.py           # 数据处理：预览、映射、统一数据生成
  data_mgmt.py          # 数据管理：详情、重命名、删除
  train_utils.py        # 后台训练任务、可取消、断点续训
  inference_utils.py    # 推理加载、显存管理、流式对话
  tb_utils.py           # TensorBoard 生命周期
models.json             # 模型列表（UI 自动维护，也可手写）
datasets_config/        # 数据集配置（数据处理自动生成）
local_data/             # 示例数据、上传文件、处理产物（git 忽略产物）
```

---

## 📄 License

本项目采用 [MIT License](LICENSE) 授权。
