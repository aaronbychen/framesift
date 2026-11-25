import asyncio
import base64
import json

import httpx

from scanner.vllm_backend import VllmSubmit


def test_request_shape_and_parsing():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "A yellow taxi."}}],
            "usage": {"prompt_tokens": 6460, "completion_tokens": 5}})

    sub = VllmSubmit()
    sub.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    text, usage = asyncio.run(sub.submit(b"\xff\xd8jpeg", "Describe.", 64))

    assert text == "A yellow taxi."
    assert usage == {"prompt_tokens": 6460, "completion_tokens": 5, "batch_size": None}
    assert seen["url"].endswith("/v1/chat/completions")
    body = seen["body"]
    assert body["temperature"] == 0 and body["max_tokens"] == 64
    image, text_part = body["messages"][0]["content"]
    assert image["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()
    assert text_part == {"type": "text", "text": "Describe."}


def test_http_error_raises():
    sub = VllmSubmit()
    sub.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, json={})))
    try:
        asyncio.run(sub.submit(b"x", "p", 8))
    except httpx.HTTPStatusError:
        return
    raise AssertionError("expected HTTPStatusError")
