# 轻量单卡训练镜像（CUDA 12.1 示例）。构建时不装模型，运行时挂载 outputs/logs。
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt requirements.txt ./
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements-web.txt -r requirements.txt

COPY . .

EXPOSE 7861 6006
ENV UNSLOTH_WEB_HOST=0.0.0.0 UNSLOTH_WEB_PORT=7861
CMD ["python3", "-m", "web", "--host", "0.0.0.0", "--port", "7861"]
