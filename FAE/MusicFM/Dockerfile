FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
RUN python3 -m pip install numpy torch torchaudio torchcodec einops transformers

ENV NVIDIA_VISIBLE_DEVICES=all
