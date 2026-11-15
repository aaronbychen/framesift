"""1-FPS frame sampling and adaptive OpenCV preprocessing."""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from scanner import config


@dataclass
class FrameStats:
    mean_luma: float
    dark_clip: float
    bright_clip: float
    noise_sigma: float
    low_light: bool
    high_contrast: bool
    noisy: bool

    @property
    def difficult(self):
        return self.low_light or self.high_contrast

    def to_dict(self):
        d = asdict(self)
        d["difficult"] = self.difficult
        return d


def estimate_noise_sigma(gray):
    """Immerkaer's fast noise variance estimate (1996) on a grayscale uint8 image."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)[1:-1, 1:-1]
    return float(np.abs(conv).sum() * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def active_area(y, black=24):
    """Crop letterbox/pillarbox bars: outer rows/cols whose brightest pixel is below `black`."""
    rows = np.where(y.max(axis=1) >= black)[0]
    cols = np.where(y.max(axis=0) >= black)[0]
    if rows.size == 0 or cols.size == 0:
        return y
    crop = y[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return crop if crop.size >= 0.2 * y.size else y


def frame_stats(bgr):
    y = active_area(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    mean_luma = float(y.mean())
    dark, bright = float((y <= 10).mean()), float((y >= 245).mean())
    sigma = estimate_noise_sigma(y)
    return FrameStats(
        mean_luma=round(mean_luma, 2),
        dark_clip=round(dark, 4),
        bright_clip=round(bright, 4),
        noise_sigma=round(sigma, 3),
        low_light=mean_luma < config.LOW_LIGHT_LUMA,
        high_contrast=min(dark, bright) >= config.HIGH_CONTRAST_CLIP_FRAC,
        noisy=sigma > config.DENOISE_SIGMA,
    )


def gray_world(bgr):
    """Gray-world color normalization: scale each channel so channel means are equal."""
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-3)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def clahe_l(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=(config.CLAHE_TILE, config.CLAHE_TILE))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def enhance(bgr, stats):
    """Denoise (when noisy) -> color normalization -> CLAHE. Denoising runs first so CLAHE
    does not amplify sensor noise in dark regions."""
    out = bgr
    if stats.noisy:
        h = float(np.clip(stats.noise_sigma * 1.2, 3, 15))
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, 21)
    out = gray_world(out)
    return clahe_l(out)


def preprocess(bgr, mode=None):
    """Returns (frame, stats, applied)."""
    mode = mode or config.PREPROCESS_MODE
    stats = frame_stats(bgr)
    if mode == "off":
        return bgr, stats, False
    if mode == "always" or (mode == "adaptive" and stats.difficult):
        return enhance(bgr, stats), stats, True
    return bgr, stats, False


def resize_for_model(bgr, max_side=None):
    max_side = max_side or config.MODEL_MAX_SIDE
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return bgr
    return cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(bgr, quality=95):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def sample_1fps(video_path):
    """Yield (second, bgr_frame) for the frame closest to t = 0, 1, 2, ... seconds.

    Uses the container's float frame rate, so a 29.97 fps clip is sampled at exactly 1 FPS
    (no int-truncation drift). Falls back to decode timestamps if the rate is missing.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        idx, second = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if fps > 0:
                target = round(second * fps)
                hit = idx == target
            else:
                hit = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= second
            if hit:
                yield second, frame
                second += 1
            idx += 1
    finally:
        cap.release()


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"fps": fps, "frames": n, "duration_s": n / fps if fps else None,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()
