# Phase 3 Completion Checklist

## A. Style foundation
- [ ] Style name is configurable
- [ ] Positive prompt is configurable
- [ ] Negative prompt is configurable
- [ ] Strength/guidance/steps are configurable

## B. Rendering
- [ ] Deterministic OpenCV renderer works
- [ ] Renderer interface is independent of video orchestration
- [ ] Optional Diffusers renderer loads a compatible model

## C. Control
- [ ] Edge control image can be produced
- [ ] Pose/hand/face structural control can be produced
- [ ] Phase 2 JSONL is consumed without modifying the Phase 2 schema

## D. Reference workflow
- [ ] Reference image can be supplied
- [ ] Reference path is passed through rendering context
- [ ] Reference conditioning can later be upgraded to IP-Adapter/LoRA/ControlNet

## E. Temporal
- [ ] Previous rendered frame is tracked
- [ ] Small changes are stabilized
- [ ] Large changes are not blindly blended
- [ ] Scene-cut reset is available

## F. Video
- [ ] Same FPS as source
- [ ] Same output resolution
- [ ] Every input frame has a corresponding output frame
- [ ] No missing frames
- [ ] Output video is playable

## G. Quality gate
The deterministic preview must pass before a diffusion model is introduced.

Then test the diffusion renderer on:
1. neutral face
2. smile
3. talking
4. hand gesture
5. body movement
6. scene change

Do not judge temporal quality from one still image. Evaluate 5–10 second clips.
