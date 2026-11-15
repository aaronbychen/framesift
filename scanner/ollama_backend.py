"""Optional Ollama backend (INFERENCE_BACKEND=ollama), one request per frame, no batching.

Kept for machines that can serve a vision model through Ollama. Note: Ollama 0.34.x cannot load
llama3.2-vision (mllama) at all, so this path only works with other vision models.
"""
import asyncio

import ollama

from scanner import config


class OllamaSubmit:
    def __init__(self):
        self.client = ollama.Client(host=config.OLLAMA_URL)

    async def submit(self, jpeg, prompt, max_tokens):
        r = await asyncio.to_thread(
            self.client.chat, model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [jpeg]}],
            options={"temperature": 0, "num_predict": max_tokens})
        return r["message"]["content"], {"prompt_tokens": r.get("prompt_eval_count"),
                                         "completion_tokens": r.get("eval_count"), "batch_size": 1}
