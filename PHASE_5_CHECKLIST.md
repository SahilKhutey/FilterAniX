# Phase 5 Completion Checklist

## Audio
- [ ] Original recorded audio is preserved.
- [ ] Audio is not accidentally replaced by generated sound.
- [ ] 48 kHz output is used.
- [ ] Stereo output is used when appropriate.
- [ ] Loudness target is configurable.
- [ ] True peak target is configurable.

## Lip-sync
- [ ] Phase 2 vision can produce a lip-sync timeline.
- [ ] Mouth states are normalized.
- [ ] Short isolated states are smoothed.
- [ ] Timeline has frame index and timestamp.
- [ ] Renderer can consume viseme data.

## Composition
- [ ] Artistic video and original audio are combined.
- [ ] Optional SRT subtitles work.
- [ ] Video encoding is deterministic.
- [ ] Final output is MP4.

## Synchronization
- [ ] Source FPS is respected.
- [ ] Source audio remains the timing authority.
- [ ] No intentional A/V retiming occurs.
- [ ] Duration mismatch is reported.

## Validation
- [ ] FFprobe detects video.
- [ ] FFprobe detects audio.
- [ ] Dimensions are valid.
- [ ] A/V drift is within tolerance.
- [ ] Final MP4 can be opened by a standard player.

## Production tests

Run:
1. 10-second talking clip
2. 30-second talking clip
3. 2-minute talking clip
4. fast speech
5. silence
6. laughter
7. multiple scene cuts
8. background music
9. subtitles
10. complete 1080p export

## Final Phase 5 architecture

```text
Original Video
 ├──────────────► Original Audio
 │                     │
 ▼                     │
Phase 2 Vision         │
 │                     │
 ▼                     │
Lip-sync Timeline      │
 │                     │
 ▼                     │
Phase 3 + 4 Render     │
 │                     │
 └──────────┬──────────┘
            ▼
       Phase 5 Compose
            │
            ├── Loudness
            ├── Subtitles
            └── Encoding
            │
            ▼
      YouTube Master MP4
```
