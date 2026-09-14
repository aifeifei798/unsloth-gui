# 轻量单卡训练镜像（CUDA 12.1 示例）。构建时不装模型，运行时挂载 outputs/logs。
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# 注意：unsloth 请按官方文档在首次构建后单独安装，保证与 torch/CUDA 匹配
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

EXPOSE 7860 6006
ENV GRADIO_HOST=0.0.0.0 GRADIO_PORT=7860 TB_PORT=6006
CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "7860"]
