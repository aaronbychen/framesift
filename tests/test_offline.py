"""The retrieval path (embedding model load, indexing, search) must not open any connection
to a non-loopback address. Needs the local Redis Stack on REDIS_URL; skipped otherwise."""
import ipaddress
import socket

import pytest

redis = pytest.importorskip("redis")
from scanner import config  # noqa: E402


def _redis_up():
    try:
        redis.Redis.from_url(config.REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _redis_up(), reason="local Redis Stack not running")


@pytest.fixture
def block_egress(monkeypatch):
    attempts = []
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host in ("localhost",)
        if not loopback:
            attempts.append(address)
            raise OSError(f"egress blocked in test: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    real_getaddrinfo = socket.getaddrinfo

    def guarded_dns(host, *a, **kw):
        if host not in ("localhost", "127.0.0.1", "::1"):
            attempts.append(("dns", host))
            raise OSError(f"DNS blocked in test: {host}")
        return real_getaddrinfo(host, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_dns)
    return attempts


def test_index_and_search_make_no_external_connections(block_egress):
    from scanner.index import CaptionIndex
    ix = CaptionIndex(name="pytest_offline")
    try:
        ix.clear()
        ix.add_clip("c1", [{"t": 0, "caption": "A dog runs across a snowy field."},
                           {"t": 1, "caption": "The dog catches a red frisbee."}], {})
        ix.add_clip("c2", [{"t": 0, "caption": "A chef slices onions in a kitchen."}], {})
        results, timings = ix.search("dog playing frisbee", k=2)
        assert results[0]["clip_id"] == "c1"
        assert results[0]["best_t"] == 1
        assert [f["t"] for f in ix.frames_between("c1", 0, 1)] == ["0", "1"]
    finally:
        ix.clear()
    assert block_egress == [], f"external connection attempts: {block_egress}"


def test_visual_retriever_fuses_and_finds_visual_only_clips(block_egress, monkeypatch):
    """Clips indexed with CLIP vectors only (captions=false) are found through the visual retriever,
    and best_t comes from the best-matching frame. A stub encoder stands in for CLIP."""
    import numpy as np
    from scanner import visual
    from scanner.index import CaptionIndex

    def unit(i):
        v = np.zeros(visual.VISUAL_DIMS, np.float32)
        v[i] = 1.0
        return v

    class Stub:
        def embed_text(self, text):
            return unit(0) if "red" in text else unit(1)

    monkeypatch.setattr(visual, "get_encoder", lambda: Stub())
    ix = CaptionIndex(name="pytest_visual")
    try:
        ix.clear()
        ix.add_clip("cap", [{"t": 0, "caption": "A chef slices onions in a kitchen."}], {})
        ix.add_visual("cap", [0], [unit(1)])
        ix.add_clip("vis", [{"t": 0, "caption": ""}, {"t": 1, "caption": ""}], {})  # CLIP-only clip
        ix.add_visual("vis", [0, 1], [unit(2), unit(0)])
        visual_only, timings = ix.search("a red car", k=2, fusion={"w_frame": 0.0})
        assert visual_only[0]["clip_id"] == "vis" and visual_only[0]["best_t"] == 1
        assert "visual_ms" in timings
        fused, _ = ix.search("a red car", k=2)  # default weights: caption 1 + visual 1
        assert {r["clip_id"] for r in fused} == {"cap", "vis"}
        caption_only, _ = ix.search("a red car", k=2, fusion={"w_visual": 0.0})
        assert all(r["clip_id"] != "vis" for r in caption_only)
        ix.delete_clip("vis")
        assert not list(ix.r.scan_iter(match="pytest_visual/vis_vis::*"))
    finally:
        ix.clear()
    assert block_egress == [], f"external connection attempts: {block_egress}"
