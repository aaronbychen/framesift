"""vLLM backend (INFERENCE_BACKEND=vllm): frames go to a vLLM OpenAI-compatible server.

Tested with vLLM 0.10.0 (V0 engine, VLLM_USE_V1=0) serving the same Llama 3.2 11B Vision
bnb-4bit checkpoint as the in-process engine; see the README for the serve command.
Greedy decoding, same prompt and token budget as the in-process engine.
"""
import base64

import httpx

from scanner import config


class VllmSubmit:
    def __init__(self):
        self.url = config.VLLM_URL.rstrip("/") + "/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=10))

    async def submit(self, jpeg, prompt, max_tokens):
        b64 = jpeg if isinstance(jpeg, str) else base64.b64encode(jpeg).decode()
        r = await self.client.post(self.url, json={
            "model": config.VLLM_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}]}]})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return d["choices"][0]["message"]["content"], {"prompt_tokens": u.get("prompt_tokens"),
                                                       "completion_tokens": u.get("completion_tokens"),
                                                       "batch_size": None}
