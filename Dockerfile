# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- OS / Python ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python3", "-m", "trade.main"]
