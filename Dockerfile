# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN --mount=type=cache,target=/model_cache \
    if [ ! -d "/model_cache/Dolphin-v2" ]; then \
        git lfs install && \
        git clone https://huggingface.co/ByteDance/Dolphin-v2 /model_cache/Dolphin-v2; \
    fi && \
    cp -r /model_cache/Dolphin-v2 ./hf_model

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
