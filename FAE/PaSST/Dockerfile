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
RUN python3 -m pip install pre-commit pytest pytest-cov pytest-env torch torchaudio torchcodec timm
#RUN python3 -m pip install av>=10.0.0 h5py>=3.8.0 jsonpickle>=3.0.1 kk=sacred>=0.8.4 librosa>=0.10.0.post2 torchmetrics>=0.2.0 pytorch-lightning<2.0.0 wandb>=0.14.2

ENV NVIDIA_VISIBLE_DEVICES=all
