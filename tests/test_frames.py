import cv2
import numpy as np
import pytest

from scanner import frames


def write_video(path, fps, n_frames, size=(160, 120)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 128, np.uint8)
        cv2.putText(img, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(img)
    vw.release()


@pytest.mark.parametrize("fps,n,expected", [(29.97, 408, 14), (25, 250, 10), (30, 31, 2)])
def test_sample_1fps_is_exactly_one_per_second(tmp_path, fps, n, expected):
    p = tmp_path / "v.mp4"
    write_video(p, fps, n)
    got = [s for s, _ in frames.sample_1fps(p)]
    assert got == list(range(expected))


def test_dark_frame_is_flagged_and_enhanced_only_in_adaptive_mode():
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(25, 8, (240, 320, 3)), 0, 255).astype(np.uint8)
    normal = np.clip(rng.normal(120, 30, (240, 320, 3)), 0, 255).astype(np.uint8)

    out, stats, applied = frames.preprocess(dark, "adaptive")
    assert stats.low_light and applied
    assert out.mean() > dark.mean()  # CLAHE + gray world brighten a dark frame

    out, stats, applied = frames.preprocess(normal, "adaptive")
    assert not stats.difficult and not applied
    assert out is normal

    assert frames.preprocess(dark, "off")[2] is False
    assert frames.preprocess(normal, "always")[2] is True


def test_letterbox_bars_do_not_count_as_low_light():
    img = np.full((240, 320, 3), 130, np.uint8)
    img[:60] = 0
    img[-60:] = 0
    stats = frames.frame_stats(img)
    assert not stats.low_light and not stats.high_contrast


def test_noise_estimate_tracks_added_noise():
    rng = np.random.default_rng(1)
    clean = np.full((200, 200), 120, np.uint8)
    noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)
    assert frames.estimate_noise_sigma(clean) < 0.5
    assert 7 < frames.estimate_noise_sigma(noisy) < 13


def test_gray_world_equalizes_channel_means():
    img = np.zeros((50, 50, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 60, 120, 180
    m = frames.gray_world(img).reshape(-1, 3).mean(axis=0)
    assert np.ptp(m) < 2
