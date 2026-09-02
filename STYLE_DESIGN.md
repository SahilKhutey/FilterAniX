# Creator Anime Style Design

The visual target should be treated as a repeatable style specification, not as a one-off prompt.

## Character

- youthful illustrated proportions
- expressive large eyes
- clean dark line work
- controlled facial geometry
- simplified but recognizable hair silhouette
- consistent clothing shapes
- warm skin rendering

## Lighting

- warm key light
- soft practical lights
- moderate shadow
- cinematic depth
- restrained highlights

## Environment

Keep real scene composition whenever possible:

- desk
- microphone
- laptop
- shelves
- plants
- lights
- wall decorations

The objective is not to replace every object randomly. It is to convert the real environment into a coherent illustrated environment.

## Motion

Preserve:

- head direction
- shoulder motion
- arm position
- hand gestures
- approximate facial expression
- camera composition

Do not allow style generation to invent a new pose.

## Identity consistency

Use a reference sheet and a fixed character description.

A production implementation should eventually combine:

- reference-image conditioning
- ControlNet pose/edge conditioning
- optional character LoRA
- temporal consistency
- face/identity protection

## Prompt template

Base:

clean cinematic anime illustration, consistent creator character,
preserve composition, preserve pose, expressive face,
clean line art, soft cel shading, warm studio lighting

Scene variables can be appended from Phase 2.

## Negative prompt

photorealistic, live action, deformed hands, extra fingers,
extra limbs, duplicate person, malformed face, inconsistent character,
text, watermark, logo, blurry, noisy
