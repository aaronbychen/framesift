"""CLIP visual encoder for the visual retriever (CLIP ViT-L/14, CPU by default).

Same model, processor and preprocessing as the v2 evaluation (eval/visual.py): raw 1-FPS frames,
standard CLIP processor (224 px, center crop), unit-normalised embeddings, query text as-is.
"""
import os
import threading
from pathlib import Path

import numpy as np

CLIP_MODEL = os.getenv("CLIP_MODEL", ".data/models/clip-vit-large-patch14")
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cpu")
VISUAL_DIMS = 768  # CLIP ViT-L/14 projection size

_lock = threading.Lock()
_encoder = None


class VisualEncoder:
    def __init__(self, path=CLIP_MODEL, device=CLIP_DEVICE):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(path).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(path)
        self._lock = threading.Lock()

    def embed_images(self, bgr_frames, batch=32):
        import cv2
        out = []
        with self._lock, self.torch.no_grad():
            for i in range(0, len(bgr_frames), batch):
                rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames[i:i + batch]]
                px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
                e = self.model.get_image_features(pixel_values=px)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, VISUAL_DIMS), np.float32)

    def embed_text(self, text):
        with self._lock, self.torch.no_grad():
            tok = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            e = self.model.get_text_features(**tok)
            return self.torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy().astype(np.float32)


def get_encoder():
    """Shared encoder, or None when the CLIP weights are not present (visual retrieval is then skipped)."""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                if not (Path(CLIP_MODEL) / "config.json").exists():
                    return None
                _encoder = VisualEncoder()
    return _encoder
