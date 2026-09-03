"""CPU Performance Benchmark for FilterAniX Mathematical Anime Engine v1.0."""
from __future__ import annotations

import time
import numpy as np

from src.art.mathematical import (
    MathematicalAnimeEngine,
    MathematicalAnimeStyle,
    compute_color_field,
    compute_tone_field,
    compute_palette_projection,
    compute_shadow_field,
    compute_highlight_field,
    compute_edge_field,
    compute_surface_normals,
    compute_lighting_field,
    compute_face_mask,
    apply_face_modulation,
    compute_foreground_mask,
    apply_background_simplification,
)


def benchmark_fields(h: int, w: int, iterations: int = 20):
    style = MathematicalAnimeStyle()
    dummy_rgb = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)

    print(f"\n--- Benchmarking Individual Fields at {w}x{h} ({iterations} runs) ---")

    # Warmup
    color_f, smooth = compute_color_field(dummy_rgb, style)
    Y_orig, Y_anime, toned = compute_tone_field(color_f, style)

    # 1. Color Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        color_f, smooth = compute_color_field(dummy_rgb, style)
    t_color = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  1. Color Field (Bilateral + Detail) : {t_color:.2f} ms")

    # 2. Tone Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        Y_orig, Y_anime, toned = compute_tone_field(color_f, style)
    t_tone = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  2. Tone Field (Luminance + S-Curve) : {t_tone:.2f} ms")

    # 3. Palette Projection
    t0 = time.perf_counter()
    for _ in range(iterations):
        pal_proj = compute_palette_projection(toned, style)
    t_pal = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  3. Palette Projection (Softmax)     : {t_pal:.2f} ms")

    # 4. Shadow Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        shaded, s_mask = compute_shadow_field(pal_proj, Y_anime, style)
    t_shadow = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  4. Cel-Shadow Field (Sigmoid)       : {t_shadow:.2f} ms")

    # 5. Highlight Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        lit, h_mask = compute_highlight_field(shaded, Y_anime, style)
    t_high = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  5. Highlight Field (Warm Tint)      : {t_high:.2f} ms")

    # 6. Edge / Ink Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        inked, i_mask, edges = compute_edge_field(lit, Y_anime, style)
    t_edge = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  6. Anime Edge / Ink Field (Sobel+L) : {t_edge:.2f} ms")

    # 7. Lighting Field
    t0 = time.perf_counter()
    for _ in range(iterations):
        lit_c, key_l = compute_lighting_field(inked, Y_anime, style)
    t_light = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  7. Cinematic Lighting Field         : {t_light:.2f} ms")

    # 8. Background Simplification
    fg_mask = compute_foreground_mask(h, w, None)
    t0 = time.perf_counter()
    for _ in range(iterations):
        simplified = apply_background_simplification(lit_c, fg_mask, style)
    t_bg = (time.perf_counter() - t0) / iterations * 1000.0
    print(f"  8. Background Simplification        : {t_bg:.2f} ms")


def benchmark_engine(h: int, w: int, frame_count: int = 30):
    style = MathematicalAnimeStyle()
    engine = MathematicalAnimeEngine(style)
    dummy_rgb = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)

    print(f"\n--- Benchmarking End-to-End Engine at {w}x{h} ({frame_count} consecutive frames) ---")

    # Warmup
    engine.render(dummy_rgb, stabilize=False)
    engine.reset()

    t0 = time.perf_counter()
    for i in range(frame_count):
        # Add slight frame perturbation to simulate natural motion
        frame = np.clip(dummy_rgb.astype(np.int16) + np.random.randint(-3, 4, dummy_rgb.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        engine.render(frame, stabilize=True)

    total_sec = time.perf_counter() - t0
    fps = frame_count / total_sec
    avg_latency = (total_sec / frame_count) * 1000.0

    print(f"  Total time       : {total_sec:.3f} s")
    print(f"  Average FPS      : {fps:.2f} FPS")
    print(f"  Average latency  : {avg_latency:.2f} ms/frame")
    summary = engine.diagnostics.summarize()
    print(f"  P95 latency      : {summary.p95_latency_ms:.2f} ms/frame")


def main():
    print("==================================================")
    print("  FilterAniX Mathematical Anime Engine Benchmark  ")
    print("==================================================")

    # 720p (Target reference resolution: 1280x720)
    benchmark_fields(720, 1280, iterations=10)
    benchmark_engine(720, 1280, frame_count=20)


if __name__ == "__main__":
    main()
