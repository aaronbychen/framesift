"""Llama 3.2 11B Vision (4-bit NF4) served in-process, behind a dynamic batching queue.

`VisionEngine.generate(batch)` runs ONE padded `model.generate` call over every item in the
batch. `BatchDispatcher` sits in front of it: concurrent requests (frames from any number of
in-flight HTTP requests) are queued, grouped into batches of up to `max_batch` items, waiting at
most `window_ms` for a partial batch to fill, and each batch is one forward pass per decode step.

With max_batch=1 the same engine reproduces the original per-request behavior: one frame per
model call, strictly in arrival order.
"""
import asyncio
import io
import os
import threading
import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram

from scanner import config

MODEL_PATH = os.getenv("VLM_PATH", ".data/models/llama32-11b-vision-bnb4")
MAX_TILES = int(os.getenv("VLM_MAX_TILES", "4"))  # the checkpoint's tile embedding is fixed at 4

BATCH_SIZE = Histogram("scanner_batch_size", "Items per model.generate call",
                       buckets=(1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32))
BATCH_SECONDS = Histogram("scanner_batch_seconds", "Wall time of one batched generate call",
                          buckets=(0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60))
QUEUE_WAIT = Histogram("scanner_queue_wait_seconds", "Time an item waits before its batch starts",
                       buckets=(0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300))
QUEUE_DEPTH = Gauge("scanner_queue_depth", "Items waiting for a batch slot")
ITEMS_TOTAL = Counter("scanner_items_total", "Items through the engine", ["status"])


@dataclass
class Item:
    image_jpeg: bytes
    prompt: str
    max_tokens: int
    future: asyncio.Future = None
    enqueued: float = field(default_factory=time.perf_counter)


class VisionEngine:
    """Owns the model; `generate` must only be called from one thread at a time."""

    def __init__(self, path=MODEL_PATH, max_tiles=MAX_TILES):
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "left"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            path, device_map="cuda:0", torch_dtype=torch.bfloat16,
            # Without this the text/cross-attention loads as "eager": ~5x slower prefill and ~6x slower decode.
            attn_implementation="sdpa")
        self.model.eval()
        self.max_tiles = max_tiles
        self.model_name = "Llama-3.2-11B-Vision-Instruct (bnb nf4, transformers)"
        self.lock = threading.Lock()

    def _image(self, jpeg):
        from PIL import Image
        return Image.open(io.BytesIO(jpeg)).convert("RGB")

    def generate(self, items):
        """items: list of (jpeg_bytes, prompt, max_tokens). Returns list of (text, usage)."""
        torch = self.torch
        texts, images = [], []
        for jpeg, prompt, _ in items:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True))
            images.append([self._image(jpeg)])
        max_new = max(m for _, _, m in items)
        with self.lock, torch.inference_mode():
            inputs = self.processor(images=images, text=texts, padding=True, add_special_tokens=False,
                                    max_image_tiles=self.max_tiles, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                      temperature=None, top_p=None,
                                      pad_token_id=self.processor.tokenizer.pad_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        attn = inputs["attention_mask"]
        results = []
        eos = set(self.model.generation_config.eos_token_id
                  if isinstance(self.model.generation_config.eos_token_id, list)
                  else [self.model.generation_config.eos_token_id])
        for i, (_, _, m) in enumerate(items):
            new = out[i, prompt_len:].tolist()
            # Truncate to this item's own token budget and at its first EOS (greedy decoding makes
            # this identical to having generated with that budget alone).
            trimmed = []
            for tok in new[:m]:
                if tok in eos or tok == self.processor.tokenizer.pad_token_id:
                    break
                trimmed.append(tok)
            text = self.processor.tokenizer.decode(trimmed, skip_special_tokens=True)
            results.append((text, {"prompt_tokens": int(attn[i].sum()), "completion_tokens": len(trimmed)}))
        return results


class BatchDispatcher:
    def __init__(self, engine, max_batch, window_ms):
        self.engine = engine
        self.max_batch = max(1, max_batch)
        self.window = window_ms / 1000.0
        self.queue = None
        self.task = None

    def start(self):
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self._run())

    async def submit(self, image_jpeg, prompt, max_tokens):
        item = Item(image_jpeg, prompt, max_tokens, asyncio.get_running_loop().create_future())
        await self.queue.put(item)
        QUEUE_DEPTH.set(self.queue.qsize())
        return await item.future

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.window
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    # Window closed: still take anything already queued, without waiting.
                    while len(batch) < self.max_batch and not self.queue.empty():
                        batch.append(self.queue.get_nowait())
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    continue
            QUEUE_DEPTH.set(self.queue.qsize())
            batch = [it for it in batch if not it.future.cancelled()]
            if not batch:
                continue
            start = time.perf_counter()
            for it in batch:
                QUEUE_WAIT.observe(start - it.enqueued)
            BATCH_SIZE.observe(len(batch))
            try:
                results = await asyncio.to_thread(
                    self.engine.generate, [(it.image_jpeg, it.prompt, it.max_tokens) for it in batch])
                for it, (text, usage) in zip(batch, results):
                    usage = dict(usage, batch_size=len(batch),
                                 queue_wait_s=round(start - it.enqueued, 4))
                    if not it.future.done():
                        it.future.set_result((text, usage))
                ITEMS_TOTAL.labels("ok").inc(len(batch))
            except Exception as e:  # noqa: BLE001 - surface the error to every caller in the batch
                ITEMS_TOTAL.labels("error").inc(len(batch))
                for it in batch:
                    if not it.future.done():
                        it.future.set_exception(e)
            finally:
                BATCH_SECONDS.observe(time.perf_counter() - start)
