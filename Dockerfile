# 设置基础镜像为官方 NVIDIA CUDA 镜像
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# 安装必要工具
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && apt-get clean

# 更新 pip
RUN pip3 install --upgrade pip

# 安装 PyTorch 和 torchvision
RUN pip3 install --no-cache-dir torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 设置工作目录
WORKDIR /MARL_project

# 复制 requirements.txt 到工作目录
COPY requirements.txt .

# 安装其他依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY train_test/ ./train_test

ENV PYTHONPATH="${PYTHONPATH}:/MARL_project"

# 设置容器启动命令
CMD ["python3", "./train_test/docker_ok.py"]
