# Phase 2 Completion Checklist

## 1. Environment
- [ ] Python 3.11+ works
- [ ] Virtual environment created
- [ ] requirements installed
- [ ] OpenCV imports
- [ ] MediaPipe imports

## 2. Face
- [ ] Face is detected in a clear talking-head frame
- [ ] Face bounding box is generated
- [ ] Face landmarks are generated
- [ ] Multiple faces do not crash the pipeline

## 3. Pose
- [ ] Body landmarks appear when upper/full body is visible
- [ ] Pose absence is handled cleanly

## 4. Hands
- [ ] One hand can be detected
- [ ] Two hands can be detected
- [ ] Missing hands do not crash processing

## 5. Segmentation
- [ ] Person segmentation statistics are produced
- [ ] No-person frames are handled

## 6. Motion
- [ ] First frame reports invalid/no previous motion
- [ ] Subsequent frames produce optical-flow statistics
- [ ] Moving video produces non-zero motion

## 7. Objects
- [ ] Core pipeline works with object detector disabled
- [ ] Optional YOLO detector can be installed
- [ ] YOLO detections map into the common Detection schema

## 8. Video
- [ ] 10–30 second video completes
- [ ] JSONL contains one record per readable frame
- [ ] summary.json is generated
- [ ] annotated.mp4 is playable

## 9. Quality Gate
Do not proceed to Phase 3 until:
- vision JSON is stable
- timestamps/frame indices are correct
- annotated video is temporally aligned
- failures are visible instead of silently ignored

## Expected Phase 2 output

For each frame:

frame
→ face
→ pose
→ hands
→ person mask statistics
→ motion
→ objects
→ timestamp

This becomes the contract for the Phase 3 Style Engine.
