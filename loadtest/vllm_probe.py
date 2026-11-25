"""Quick vLLM concurrency probe: first 8 frames of uploads/test.mp4 at 1120 px and 560 px, captioned
one at a time and then all at once against a running vLLM server on :8100.

    python loadtest/vllm_probe.py artifacts/loadtest/vllm_concurrency_probe.json
"""
import asyncio, base64, json, sys, time
import httpx
sys.path.insert(0, ".")
from scanner.frames import sample_1fps, preprocess, resize_for_model, encode_jpeg
from scanner.pipeline import CAPTION_PROMPT
raw = [f for _, f in sample_1fps("uploads/test.mp4")][:8]

def body(jpeg):
    b64 = jpeg if isinstance(jpeg, str) else base64.b64encode(jpeg).decode()
    return {"model": "llama32-vision", "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": CAPTION_PROMPT}]}]}

async def one(c, j):
    t = time.perf_counter()
    r = await c.post("http://127.0.0.1:8100/v1/chat/completions", json=body(j))
    if r.status_code != 200: raise SystemExit(r.text[:400])
    d = r.json()
    return time.perf_counter() - t, d["usage"], d["choices"][0]["message"]["content"]

async def run(side):
    frames = [encode_jpeg(resize_for_model(preprocess(f, "off")[0], side)) for f in raw]
    async with httpx.AsyncClient(timeout=900) as c:
        await one(c, frames[0])
        t = time.perf_counter(); seq = [await one(c, j) for j in frames]; ts = time.perf_counter() - t
        t = time.perf_counter(); par = await asyncio.gather(*(one(c, j) for j in frames)); tp = time.perf_counter() - t
    toks = sum(x[1]["completion_tokens"] for x in seq)
    print(f"[{side}px] prompt_tokens={seq[0][1]['prompt_tokens']} | sequential {len(frames)/ts:.3f} fps ({ts/len(frames):.2f} s/frame, {toks/sum(x[0] for x in seq):.1f} tok/s) "
          f"| 8 concurrent {len(frames)/tp:.3f} fps (x{ts/tp:.2f}), mean latency {sum(x[0] for x in par)/8:.2f}s | same text {sum(a[2]==b[2] for a,b in zip(seq,par))}/8", flush=True)
    return {"side": side, "sequential_s": ts, "concurrent_s": tp, "seq": [x[2] for x in seq], "par": [x[2] for x in par]}

async def main():
    out = [await run(s) for s in (1120, 560)]
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("1120 sample:", out[0]["seq"][0][:150]); print(" 560 sample:", out[1]["seq"][0][:150])
asyncio.run(main())
