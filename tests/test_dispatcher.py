import asyncio
import time

from scanner.vlm import BatchDispatcher


class FakeEngine:
    def __init__(self, delay=0.05):
        self.calls = []
        self.delay = delay

    def generate(self, items):
        self.calls.append(len(items))
        time.sleep(self.delay)
        return [(f"out:{prompt}", {"completion_tokens": 1}) for _, prompt, _ in items]


async def run(engine, max_batch, window_ms, n, stagger=0.0):
    d = BatchDispatcher(engine, max_batch, window_ms)
    d.start()

    async def one(i):
        await asyncio.sleep(i * stagger)
        return await d.submit(b"jpeg", str(i), 8)

    results = await asyncio.gather(*(one(i) for i in range(n)))
    d.task.cancel()
    return results


def test_concurrent_requests_are_grouped_into_batches():
    eng = FakeEngine()
    results = asyncio.run(run(eng, max_batch=4, window_ms=50, n=8))
    assert [r[0] for r in results] == [f"out:{i}" for i in range(8)]  # each caller gets its own answer
    assert eng.calls == [4, 4]
    assert all(r[1]["batch_size"] == 4 for r in results)


def test_max_batch_one_is_strictly_one_item_per_call():
    eng = FakeEngine(delay=0.01)
    asyncio.run(run(eng, max_batch=1, window_ms=50, n=5))
    assert eng.calls == [1] * 5


def test_partial_batch_is_flushed_after_window():
    eng = FakeEngine(delay=0.0)
    start = time.perf_counter()
    asyncio.run(run(eng, max_batch=8, window_ms=30, n=3))
    assert eng.calls == [3]
    assert time.perf_counter() - start < 1.0


def test_engine_error_reaches_every_caller_in_the_batch():
    class Boom(FakeEngine):
        def generate(self, items):
            raise RuntimeError("cuda oom")

    async def go():
        d = BatchDispatcher(Boom(), 4, 20)
        d.start()
        res = await asyncio.gather(*(d.submit(b"x", "p", 1) for _ in range(3)), return_exceptions=True)
        d.task.cancel()
        return res

    res = asyncio.run(go())
    assert all(isinstance(r, RuntimeError) for r in res)
