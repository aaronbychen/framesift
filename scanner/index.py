"""Offline retrieval layer: LlamaIndex over a local Redis (NoSQL) caption + frame/time index.

Redis layout (one local redis-stack-server, nothing leaves the machine):
  clip:{clip_id}            hash   clip metadata (path, duration, fps, frame count, preprocessing)
  frames:{clip_id}          zset   frame/time index: member "t", score t (seconds)
  frame:{clip_id}:{t}       hash   per-frame caption + lighting stats + whether it was enhanced
  {INDEX_NAME}/cap_*        hash   LlamaIndex RedisVectorStore nodes (HNSW, cosine), one per
                                   frame caption (kind=frame) and one per clip (kind=clip)

Query path (no LLM involved, so it runs fully offline):
  1. dense frame retrieval  - top-N frame captions, clip score = best frame (gives a timestamp)
  2. dense clip retrieval   - top-N clip-level caption documents
  3. BM25 clip retrieval    - lexical match over the same clip documents
  4. weighted reciprocal-rank fusion at clip level -> top-k clips, each with its best timestamp
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
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

from scanner import config

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))
# Fusion settings, tuned on the MSR-VTT train-split dev set only (see eval/README.md).
FUSION = {
    "rrf_k": float(os.getenv("RRF_K", "60")),
    "w_frame": float(os.getenv("W_FRAME", "1.0")),
    "w_clip": float(os.getenv("W_CLIP", "1.0")),
    "w_bm25": float(os.getenv("W_BM25", "1.0")),
    "candidates": int(os.getenv("FUSION_CANDIDATES", "50")),
    "frame_candidates": int(os.getenv("FRAME_CANDIDATES", "200")),
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
        self.index.insert_nodes(nodes)

    def delete_clip(self, clip_id):
        ts = self.r.zrange(self.k("frames", clip_id), 0, -1)
        keys = [self.k("frame", clip_id, t) for t in ts] + [self.k("frames", clip_id), self.k("clip", clip_id)]
        node_keys = list(self.r.scan_iter(match=f"{self.name}/cap_{clip_id}::*", count=1000))
        if keys or node_keys:
            self.r.delete(*keys, *node_keys)
        self.r.srem(self.k("clips"), clip_id)
        self.r.incr(self.k("version"))

    def clear(self):
        for key in self.r.scan_iter(match=f"{self.name}:*", count=1000):
            self.r.delete(key)
        for key in self.r.scan_iter(match=f"{self.name}/cap_*", count=1000):
            self.r.delete(key)

    # ---------- reads ----------
    def clip_ids(self):
        return sorted(self.r.smembers(self.k("clips")))

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
            results.append({"clip_id": cid, "score": round(s, 6), "best_t": t, "best_caption": cap})
        return results, {key: round(v, 3) for key, v in timings.items()}
