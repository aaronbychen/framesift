# vLLM 0.10.0 for Llama 3.2 11B Vision (mllama) on a CUDA 12.6 driver.
# - 0.10.0 is the newest vLLM with a +cu126 wheel; the host driver (566.07) supports CUDA <= 12.7.
# - mllama only runs on vLLM's V0 engine (removed in 0.11.0): serve with VLLM_USE_V1=0.
# - bitsandbytes 0.46.1 is the minimum vLLM 0.10.0 accepts for the pre-quantized NF4 checkpoint.
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# Resolve unpinned dependencies as of the vLLM 0.10.0 release date so newer releases can't break it.
RUN pip install --no-cache-dir uv \
    && uv pip install --no-cache --exclude-newer 2025-07-26T00:00:00Z \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.10.0/vllm-0.10.0+cu126-cp38-abi3-manylinux1_x86_64.whl" \
    && pip install --no-cache-dir bitsandbytes==0.46.1

RUN python -c "import torch, vllm, transformers, bitsandbytes; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'bnb', bitsandbytes.__version__); assert torch.version.cuda.startswith('12.6'), torch.version.cuda"

ENV VLLM_USE_V1=0 \
    HF_HUB_OFFLINE=1
ENTRYPOINT ["vllm", "serve"]
