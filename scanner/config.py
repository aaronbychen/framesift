"""All tunables in one place, read from the environment."""
import os


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

# Inference serving. "sequential": the original path, one model call per frame, awaited in order.
# "batched": frames from every in-flight request go through one BatchDispatcher.
SERVING_MODE = os.getenv("SERVING_MODE", "batched")
BATCH_MAX_SIZE = _i("BATCH_MAX_SIZE", 4)          # should equal the model server's parallel slots
BATCH_WINDOW_MS = _f("BATCH_WINDOW_MS", 25)       # how long a partial batch waits for more frames
# Frames per request allowed in flight at once in batched mode.
STREAM_WINDOW = _i("STREAM_WINDOW", 8)

# Frames are downscaled so the longest side is at most this before going to the model.
MODEL_MAX_SIDE = _i("MODEL_MAX_SIDE", 1120)
ANALYZE_MAX_TOKENS = _i("ANALYZE_MAX_TOKENS", 256)
CAPTION_MAX_TOKENS = _i("CAPTION_MAX_TOKENS", 64)

# Preprocessing: "off", "always", or "adaptive" (only frames flagged difficult by the thresholds).
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "adaptive")
LOW_LIGHT_LUMA = _f("LOW_LIGHT_LUMA", 50)          # mean Y (0-255) of the active area below this => low light
# High contrast = both tails clipped: at least this share of pixels at Y<=10 AND at Y>=245.
HIGH_CONTRAST_CLIP_FRAC = _f("HIGH_CONTRAST_CLIP_FRAC", 0.05)
DENOISE_SIGMA = _f("DENOISE_SIGMA", 4.0)           # estimated noise sigma above this => denoise
CLAHE_CLIP_LIMIT = _f("CLAHE_CLIP_LIMIT", 2.0)
CLAHE_TILE = _i("CLAHE_TILE", 8)

# Retrieval layer.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
INDEX_NAME = os.getenv("INDEX_NAME", "scanner")
EMBED_MODEL = os.getenv("EMBED_MODEL", ".data/models/bge-base-en-v1.5")  # local copy of BAAI/bge-base-en-v1.5
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
