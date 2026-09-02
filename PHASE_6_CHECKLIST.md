# Phase 6 Production Checklist

## Application
- [ ] Local application starts.
- [ ] Source video can be selected.
- [ ] Style can be selected.
- [ ] Project can be created.
- [ ] Project manifest is written.
- [ ] Logs are written.

## Pipeline
- [ ] Phase 1 adapter connected.
- [ ] Phase 2 adapter connected.
- [ ] Phase 3 adapter connected.
- [ ] Phase 4 adapter connected.
- [ ] Phase 5 adapter connected.
- [ ] Progress callback connected.
- [ ] Failed jobs persist errors.
- [ ] Retry works.

## Live mode
- [ ] Webcam is detected.
- [ ] Low-latency preview works.
- [ ] Fast renderer is used.
- [ ] Temporal state is persistent.
- [ ] Final quality mode is not used for every live frame.

## Hardware
- [ ] CPU detected.
- [ ] RAM detected.
- [ ] FFmpeg detected.
- [ ] GPU detected when available.
- [ ] CUDA detected when available.
- [ ] CPU fallback remains usable.

## Export
- [ ] 720p works.
- [ ] 1080p works.
- [ ] 1440p works.
- [ ] 2160p works.
- [ ] H.264 output works.
- [ ] AAC audio works.
- [ ] 48 kHz audio works.
- [ ] faststart enabled.
- [ ] Output validation passes.

## Production hardening

Before calling the application production-ready:

1. Replace Phase adapters with real implementations.
2. Add a persistent job database if multiple concurrent jobs are required.
3. Add cancellation tokens.
4. Add disk-space checks before rendering.
5. Add model checksum/version tracking.
6. Cache model instances instead of loading them per frame.
7. Limit queue concurrency according to VRAM.
8. Save checkpoints after each phase.
9. Add crash recovery.
10. Add structured JSON logs.
11. Add automatic cleanup policies for temporary frames.
12. Add a project lock to prevent two workers writing the same project.
13. Add GPU memory monitoring.
14. Add preview thumbnails.
15. Add a render-quality selector: Preview / Balanced / Final.

## Recommended quality modes

### Preview

```text
low resolution
low frame count
fast renderer
minimal diffusion steps
```

### Balanced

```text
720p/1080p
moderate steps
temporal stabilization
```

### Final

```text
source FPS
full resolution
strong identity conditioning
full temporal pass
audio composition
YouTube export
```

## Final six-phase product

```text
PHASE 1
Input Infrastructure
       ↓
PHASE 2
Vision / Scene Understanding
       ↓
PHASE 3
Artistic Style Engine
       ↓
PHASE 4
Identity / Temporal Consistency
       ↓
PHASE 5
Audio / Lip-Sync / Composition
       ↓
PHASE 6
Production Application / Export
       ↓
      YouTube
```
