"""Offline retrieval layer: LlamaIndex over a local Redis (NoSQL) caption + frame/time index.

Redis layout (one local redis-stack-server, nothing leaves the machine):
  clip:{clip_id}            hash   clip metadata (path, duration, fps, frame count, preprocessing)
  frames:{clip_id}          zset   frame/time index: member "t", score t (seconds)
  frame:{clip_id}:{t}       hash   per-frame caption + lighting stats + whether it was enhanced
  {INDEX_NAME}/cap_*        hash   LlamaIndex RedisVectorStore nodes (HNSW, cosine), one per
                                   frame caption (kind=frame) and one per clip (kind=clip)
  {INDEX_NAME}/vis_*        hash   second RedisVectorStore: one CLIP image embedding per frame

Query path (no LLM involved, so it runs fully offline):
  1. dense frame retrieval  - top-N frame captions, clip score = best frame (gives a timestamp)
  2. dense clip retrieval   - top-N clip-level caption documents
  3. BM25 clip retrieval    - lexical match over the same clip documents
  4. CLIP visual retrieval  - query text vs frame images, clip score = best frame
  5. weighted reciprocal-rank fusion at clip level -> top-k clips, each with its best timestamp
"""
import json
import os
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import redis
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

from scanner import config

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))
# Fusion weights frozen from the dev-set grid (artifacts/eval/dev80_fusion_grid.json, MSR-VTT train
# clips only): best dev Hit@5 was dense frame retrieval alone (0.900); the best fused config scored
# 0.8875. Clip-level and BM25 retrievers stay available through W_CLIP / W_BM25.
FUSION = {
    "rrf_k": float(os.getenv("RRF_K", "60")),
    "w_frame": float(os.getenv("W_FRAME", "1.0")),
    "w_clip": float(os.getenv("W_CLIP", "0.0")),
    "w_bm25": float(os.getenv("W_BM25", "0.0")),
    # CLIP visual retriever, weight frozen from the pre-specified v2 dev grid
    # (artifacts/eval/dev80_visual_grid.json: caption 1 + visual 1 selected).
    "w_visual": float(os.getenv("W_VISUAL", "1.0")),
    "candidates": int(os.getenv("FUSION_CANDIDATES", "50")),
    "frame_candidates": int(os.getenv("FRAME_CANDIDATES", "200")),
    "visual_candidates": int(os.getenv("VISUAL_CANDIDATES", "100")),
    "visual_frame_candidates": int(os.getenv("VISUAL_FRAME_CANDIDATES", "1000")),
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def make_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    kwargs = {}
    if "bge" in config.EMBED_MODEL.lower():
        kwargs["query_instruction"] = BGE_QUERY_PREFIX
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL, device=config.EMBED_DEVICE,
                                embed_batch_size=64, **kwargs)


def make_schema(name):
    return IndexSchema.from_dict({
        "index": {"name": name, "prefix": f"{name}/cap", "key_separator": "_", "storage_type": "hash"},
        "fields": [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {"type": "tag", "name": "clip_id"},
            {"type": "tag", "name": "kind"},
            {"type": "numeric", "name": "t"},
            {"type": "vector", "name": "vector",
             "attrs": {"dims": EMBED_DIMS, "algorithm": "hnsw", "distance_metric": "cosine",
                       "m": 16, "ef_construction": 200, "ef_runtime": 100}},
        ],
    })


def make_visual_schema(name):
    from scanner.visual import VISUAL_DIMS
    return IndexSchema.from_dict({
        "index": {"name": f"{name}_vis", "prefix": f"{name}/vis", "key_separator": "_", "storage_type": "hash"},
        "fields": [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {"type": "tag", "name": "clip_id"},
            {"type": "numeric", "name": "t"},
            {"type": "vector", "name": "vector",
             "attrs": {"dims": VISUAL_DIMS, "algorithm": "hnsw", "distance_metric": "cosine",
                       "m": 16, "ef_construction": 200, "ef_runtime": 200}},
        ],
    })


def clip_document(captions):
    """captions: list of (t, caption) -> one text block for clip-level retrieval."""
    return " ".join(c for _, c in sorted(captions))


class CaptionIndex:
    def __init__(self, name=None, redis_url=None, embed_model=None):
        self.name = name or config.INDEX_NAME
        self.r = redis.Redis.from_url(redis_url or config.REDIS_URL, decode_responses=True)
        Settings.llm = None
        Settings.embed_model = embed_model or make_embed_model()
        self.vector_store = RedisVectorStore(schema=make_schema(self.name), redis_url=redis_url or config.REDIS_URL,
                                             overwrite=False)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.visual_store = RedisVectorStore(schema=make_visual_schema(self.name),
                                             redis_url=redis_url or config.REDIS_URL, overwrite=False)
        self._bm25 = None
        self._bm25_version = None
        self._lock = threading.Lock()

    # ---------- writes ----------
    def k(self, *parts):
        return ":".join([self.name, *map(str, parts)])

    def add_clip(self, clip_id, frames, meta):
        """frames: list of dicts with keys t, caption, stats (dict), enhanced (bool), frame_path."""
        self.delete_clip(clip_id)
        pipe = self.r.pipeline()
        pipe.hset(self.k("clip", clip_id), mapping={
            "clip_id": clip_id, "meta": json.dumps(meta), "n_frames": len(frames), "indexed_at": time.time()})
        for f in frames:
            pipe.zadd(self.k("frames", clip_id), {str(f["t"]): f["t"]})
            pipe.hset(self.k("frame", clip_id, f["t"]), mapping={
                "t": f["t"], "caption": f["caption"], "stats": json.dumps(f.get("stats", {})),
                "enhanced": int(bool(f.get("enhanced"))), "frame_path": f.get("frame_path", "")})
        pipe.sadd(self.k("clips"), clip_id)
        pipe.incr(self.k("version"))
        pipe.execute()

        nodes = [TextNode(id_=f"{clip_id}::f{f['t']}", text=f["caption"],
                          metadata={"clip_id": clip_id, "kind": "frame", "t": f["t"]})
                 for f in frames if f["caption"]]
        doc = clip_document([(f["t"], f["caption"]) for f in frames if f["caption"]])
        if doc:
            nodes.append(TextNode(id_=f"{clip_id}::clip", text=doc,
                                  metadata={"clip_id": clip_id, "kind": "clip", "t": 0}))
        for n in nodes:
            n.excluded_embed_metadata_keys = ["clip_id", "kind", "t"]
            n.excluded_llm_metadata_keys = ["clip_id", "kind", "t"]
        if nodes:
            self.index.insert_nodes(nodes)

    def add_visual(self, clip_id, ts, embeddings):
        """One CLIP image embedding per frame (unit-norm, from scanner.visual)."""
        nodes = [TextNode(id_=f"{clip_id}::v{t}", text="", embedding=[float(x) for x in e],
                          metadata={"clip_id": clip_id, "t": t})
                 for t, e in zip(ts, embeddings)]
        if nodes:
            self.visual_store.add(nodes)
            self.r.incr(self.k("version"))

    def delete_clip(self, clip_id):
        ts = self.r.zrange(self.k("frames", clip_id), 0, -1)
        keys = [self.k("frame", clip_id, t) for t in ts] + [self.k("frames", clip_id), self.k("clip", clip_id)]
        node_keys = list(self.r.scan_iter(match=f"{self.name}/cap_{clip_id}::*", count=1000)) +             list(self.r.scan_iter(match=f"{self.name}/vis_{clip_id}::*", count=1000))
        if keys or node_keys:
            self.r.delete(*keys, *node_keys)
        self.r.srem(self.k("clips"), clip_id)
        self.r.incr(self.k("version"))

    def clear(self):
        for key in self.r.scan_iter(match=f"{self.name}:*", count=1000):
            self.r.delete(key)
        for pattern in (f"{self.name}/cap_*", f"{self.name}/vis_*"):
            for key in self.r.scan_iter(match=pattern, count=1000):
                self.r.delete(key)

    # ---------- reads ----------
    def clip_ids(self):
        return sorted(self.r.smembers(self.k("clips")))

    def clip_meta(self, clip_id):
        h = self.r.hgetall(self.k("clip", clip_id))
        if not h:
            return None
        return {"clip_id": clip_id, "n_frames": int(h.get("n_frames", 0)), "meta": json.loads(h.get("meta", "{}"))}

    def frames_between(self, clip_id, t0, t1):
        """Frame/time index lookup: captions for t0 <= t <= t1."""
        ts = self.r.zrangebyscore(self.k("frames", clip_id), t0, t1)
        return [self.r.hgetall(self.k("frame", clip_id, t)) for t in ts]

    def _bm25_retriever(self):
        version = self.r.get(self.k("version"))
        with self._lock:
            if self._bm25 is None or version != self._bm25_version:
                from llama_index.retrievers.bm25 import BM25Retriever
                nodes = []
                for cid in self.clip_ids():
                    ts = self.r.zrange(self.k("frames", cid), 0, -1)
                    caps = [(float(t), self.r.hget(self.k("frame", cid, t), "caption") or "") for t in ts]
                    nodes.append(TextNode(id_=f"{cid}::clip", text=clip_document(caps), metadata={"clip_id": cid}))
                self._bm25 = BM25Retriever.from_defaults(nodes=nodes,
                                                         similarity_top_k=min(FUSION["candidates"], len(nodes)))
                self._bm25_version = version
            return self._bm25

    def _dense(self, query_bundle, kind, top_k):
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=MetadataFilters(filters=[MetadataFilter(key="kind", value=kind, operator=FilterOperator.EQ)]))
        return retriever.retrieve(query_bundle)

    def _visual(self, query, f, best_visual):
        """CLIP text embedding vs frame image embeddings -> clip ids ranked by their best frame."""
        from scanner.visual import get_encoder
        enc = get_encoder()
        if enc is None:
            return []
        res = self.visual_store.query(VectorStoreQuery(query_embedding=enc.embed_text(query).tolist(),
                                                       similarity_top_k=f["visual_frame_candidates"]))
        seen = []
        for node in res.nodes or []:
            cid = node.metadata["clip_id"]
            if cid not in best_visual:
                t = node.metadata["t"]
                best_visual[cid] = int(t) if float(t).is_integer() else t
                seen.append(cid)
        return seen[:f["visual_candidates"]]

    def search(self, query, k=5, fusion=None):
        from llama_index.core.schema import QueryBundle
        f = dict(FUSION, **(fusion or {}))
        timings = {}
        t0 = time.perf_counter()
        qb = QueryBundle(query_str=query, embedding=Settings.embed_model.get_query_embedding(query))
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        ranked = {}
        best_frame = {}
        if f["w_frame"] > 0:
            seen = []
            for n in self._dense(qb, "frame", f["frame_candidates"]):
                cid = n.node.metadata["clip_id"]
                if cid not in best_frame:
                    best_frame[cid] = (n.node.metadata["t"], n.node.get_content(), n.score)
                    seen.append(cid)
            ranked["frame"] = seen[:f["candidates"]]
        if f["w_clip"] > 0:
            ranked["clip"] = [n.node.metadata["clip_id"] for n in self._dense(qb, "clip", f["candidates"])]
        timings["vector_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        if f["w_bm25"] > 0:
            ranked["bm25"] = [n.node.metadata["clip_id"] for n in self._bm25_retriever().retrieve(query)]
        timings["bm25_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        best_visual = {}
        if f["w_visual"] > 0:
            ranked["visual"] = self._visual(query, f, best_visual)
        timings["visual_ms"] = (time.perf_counter() - t3) * 1000

        scores = {}
        for name, ids in ranked.items():
            w = f[f"w_{name}"]
            for rank, cid in enumerate(ids):
                scores[cid] = scores.get(cid, 0.0) + w / (f["rrf_k"] + rank + 1)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        results = []
        for cid, s in top:
            t, cap, _ = best_frame.get(cid, (None, None, None))
            if t is None and cid in best_visual:  # found only visually: use the best-matching frame
                t = best_visual[cid]
                cap = self.r.hget(self.k("frame", cid, t), "caption") or None
            results.append({"clip_id": cid, "score": round(s, 6), "best_t": t, "best_caption": cap})
        return results, {key: round(v, 3) for key, v in timings.items()}
