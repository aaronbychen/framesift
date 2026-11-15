"""Frame pipeline shared by the API and the eval scripts: 1-FPS sampling -> adaptive
preprocessing -> model calls through the dispatcher, with frames streamed in order."""
import asyncio
import threading
from collections import deque
from pathlib import Path

import cv2

from scanner import config
from scanner.frames import encode_jpeg, preprocess, resize_for_model, sample_1fps

CAPTION_PROMPT = (
    "Describe this video frame in one sentence of at most 40 words for a video search index: "
    "the main people, animals or objects, what they are doing, and the setting. "
    "Be concrete. No lists, no preamble."
)


def analyze_prompt(object_str):
    return f"""Please analyze the image and answer the following questions:
    1. Is there a {object_str} in the image?
    2. If yes, describe its appearance and location in the image in detail.
    3. If no, describe what you see in the image instead.

    Format your response as:
    Answer: [YES/NO]
    Description: [Your detailed description]
    Confidence: [1-10]"""


def parse_analysis(text):
    answer, description, confidence = None, None, 10
    for line in text.strip().split("\n"):
        line = line.strip()
        low = line.lower()
        if low.startswith("answer:"):
            answer = line.split(":", 1)[1].strip().upper()
        elif any(low.startswith(p) for p in ("description:", "reasoning:", "alternative description:")):
            description = line.split(":", 1)[1].strip()
        elif low.startswith("confidence:"):
            try:
                confidence = int(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 10
    return answer == "YES" and confidence >= 7, description, confidence


async def iter_frames(video_path, preprocess_mode=None, save_dir=None, queue_size=16):
    """Async iterator of (second, jpeg_bytes, stats, enhanced, frame_path). Decoding and
    OpenCV work run in a worker thread so the event loop never blocks on them."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(queue_size)
    stop = threading.Event()
    done = object()

    def put(item):
        fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
        fut.result()

    def worker():
        try:
            for second, frame in sample_1fps(video_path):
                if stop.is_set():
                    break
                out, stats, enhanced = preprocess(frame, preprocess_mode)
                frame_path = ""
                if save_dir is not None:
                    p = Path(save_dir) / f"frame_{second}.jpg"
                    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_path = str(p)
                put((second, encode_jpeg(resize_for_model(out)), stats.to_dict(), enhanced, frame_path))
        except BaseException as e:  # noqa: BLE001 - re-raised on the event loop side
            put(e)
        finally:
            put(done)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = await q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while not q.empty():
            q.get_nowait()


async def run_frames(submit, video_path, prompt, max_tokens, window, preprocess_mode=None, save_dir=None):
    """Yield (frame_tuple, (text, usage) | Exception) in frame order.

    window == 1 reproduces the original control flow: the next frame is not submitted until the
    previous frame's model call returns. window > 1 keeps up to `window` frames of this request
    in flight, so they can share batches with each other and with other requests.
    """
    pending = deque()

    async def call(jpeg):
        try:
            return await submit(jpeg, prompt, max_tokens)
        except Exception as e:  # noqa: BLE001
            return e

    try:
        async for fr in iter_frames(video_path, preprocess_mode, save_dir):
            pending.append((fr, asyncio.create_task(call(fr[1]))))
            while pending and (len(pending) >= window or pending[0][1].done()):
                f, task = pending.popleft()
                yield f, await task
        while pending:
            f, task = pending.popleft()
            yield f, await task
    finally:
        for _, task in pending:
            task.cancel()


async def caption_clip(submit, video_path, window=None, preprocess_mode=None, save_dir=None):
    frames = []
    async for (second, _, stats, enhanced, frame_path), res in run_frames(
            submit, video_path, CAPTION_PROMPT, config.CAPTION_MAX_TOKENS,
            window or config.STREAM_WINDOW, preprocess_mode, save_dir):
        if isinstance(res, Exception):
            frames.append({"t": second, "caption": "", "error": repr(res), "stats": stats,
                           "enhanced": enhanced, "frame_path": frame_path})
        else:
            text, usage = res
            frames.append({"t": second, "caption": text.strip(), "usage": usage, "stats": stats,
                           "enhanced": enhanced, "frame_path": frame_path})
    return frames
