"""
Serve openai/gpt-oss-safeguard-20b on Modal via vLLM.

Usage:
    modal secret create cope-secrets HF_TOKEN=hf_... VLLM_API_KEY=sk-...
    modal run serve_safeguard.py::download_model    # one-time, pre-warms the volume
    modal deploy serve_safeguard.py                  # publishes the endpoint
    modal app stop gpt-oss-safeguard                 # tear down when done
"""

import modal

MODEL_NAME = "openai/gpt-oss-safeguard-20b"
GPU_CONFIG = "H100:1"
MAX_MODEL_LEN = 16384  # match cope-b so we can fit very_long policies
SCALEDOWN_SECONDS = 10 * 60

app = modal.App("gpt-oss-safeguard")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("vllm")
    .pip_install("hf-transfer", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("cope-secrets")]


@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=secrets,
    timeout=60 * 60,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    hf_cache.commit()
    print(f"Cached {MODEL_NAME} to volume.")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=secrets,
    timeout=24 * 60 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    import os
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--trust-remote-code",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
