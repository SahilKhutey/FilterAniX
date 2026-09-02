# FilterAniX Architecture Specification

FilterAniX is structured as a 6-phase sequential pipeline with independent workspaces, checkpointing, and source timing preservation.

```
                           REAL CREATOR VIDEO
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 1: Video Input & Metadata Infrastructure │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 2: Vision & Scene Understanding Engine   │
          │  (Face Mesh, 3D Pose, Hands, Mask, Flow, Props) │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 3: Artistic Style & Cel-Shading Engine   │
          │  (Procedural Kuwahara, CIELAB Cel, Inking, Flow)│
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 4: Character Identity & Temporal Engine  │
          │  (Reference Signatures, Scene Cuts, Keyframes)  │
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 5: Lip-Sync & Multi-Track Composition    │
          │  (4-State Visemes, EBU R128 Loudnorm, Voice Mux)│
          └────────────────────────┬────────────────────────┘
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  PHASE 6: Production Studio & YouTube Exporter  │
          │  (Manifest Engine, Gradio Web UI, Multi-Res MP4)│
          └────────────────────────┬────────────────────────┘
                                   ▼
                         🎬 YOUTUBE MASTER VIDEO
```

---

## Core Mathematical & Algorithmic Foundations

### 1. Extended Difference of Gaussians (XDoG) Inking
Edge extraction is formulated as:
$$D_{\sigma, k, \tau, \epsilon}(x, y) = (1 + \tau) \cdot G_\sigma(x, y) - \tau \cdot G_{k\sigma}(x, y)$$
A smooth hyperbolic tangent thresholding operator creates organic anime ink line contours:
$$T(u) = \begin{cases} 1.0 & \text{if } u \ge \epsilon \\ 1.0 + \tanh(\phi \cdot (u - \epsilon)) & \text{otherwise} \end{cases}$$

### 2. Reinhard CIELAB Reference Palette Transfer
Given a canonical reference character palette, the input frame color statistics are shifted in $L^*a^*b^*$ perceptual color space:
$$L^*_{\text{out}} = \frac{\sigma_L^{\text{ref}}}{\sigma_L^{\text{src}}} (L^*_{\text{src}} - \mu_L^{\text{src}}) + \mu_L^{\text{ref}}$$

### 3. Dense Farneback Optical Flow Temporal Stabilization
Inter-frame temporal stabilization warps prior stylized output along motion vectors:
$$I_{\text{stab}}(t) = (1 - \alpha) \cdot I_{\text{stylized}}(t) + \alpha \cdot \mathcal{W}(I_{\text{stab}}(t-1), \mathbf{v}_{t-1 \to t})$$

### 4. Audio Timing Authority & EBU R128 Broadcast Normalization
Source audio is normalized to YouTube broadcast standard:
- Target Loudness: **-14.0 LUFS**
- Maximum True Peak: **-1.5 dBTP**
- Loudness Range (LRA): **11 LU**
