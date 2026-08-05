# Phase 1: Documentation

## Problem Statement

- The current systems are enrolled and trained on full, unmasked face photos that rely heavily on lower-face features such as the nose, mouth, lips, and chin to identify people.  
  Current ArcFace baseline: **99.8%**

- In real-world conditions, people frequently wear masks that hide these lower-face features, causing the system to fail at recognizing them.

## Problems with Current Approach

| Dataset | Training Setup | Evaluation Protocol | Baseline (Unmasked) | Masked Performance |
|---|---|---|---|---|
| Masked LFW (MLFW) | ArcFace trained only on unmasked faces | 1:1 Verification, 6,000 pairs | 99.83% LFW Verification | 78.40% |
| Masked LFW (MLFW) | ArcFace trained only on unmasked faces | 1:N Identification, Rank-1 | 72.50% | 74.10% |

## Enhancements

1. Rather than just classifying identities, we will generate clean/occluded *pairs* of the same face and train the head to minimize the distance between their embeddings using MaskTheFace and 5-point alignment.

2. The head learns a *correction* on top of the frozen embedding rather than a full new representation.

3. Use a residual adapter, such as:

   ```text
   512 -> 256 -> 512
   ```

   with output:

   ```text
   z' = normalize(z + Δz)
   ```

   This preserves ArcFace identity information while correcting mask-induced embedding drift.

## Occlusion Head Training Pipeline

```text
Public face dataset
        |
        v
Detect and align
SCRFD, 5-point, 112x112
        |
        v
Augmentation
Masks, sunglasses, hands, cutouts
Blur, lighting, JPEG, pose jitter
Produces clean-occluded pairs
        |
        v
Frozen ArcFace encoder
        |
        v
Clean embedding        Occluded embedding
512-D                  512-D
        |                    |
        |                    v
        |             Residual head
        |             512 -> 256 -> 512
        |                    |
        v                    v
        Recovered embedding
        |
        v
Identity losses
Cosine + ArcFace margin
        |
        v
Export occlusion_head.onnx
```

## How This Fits in the Enrollment Pipeline

```text
One clean image
        |
        v
SCRFD detection
        |
        v
Quality check
Blur, pose, occlusion gate
        |
        v
5-point alignment
        |
        v
InsightFace ArcFace
        |
        v
Trained occlusion head
Refines the 512-D embedding
        |
        v
L2-normalized embedding
512-D unit vector
        |
        v
Gallery entry
person_id maps to embedding
```

## Dataset to Train Head

**CASIA-WebFace**: 494,414 face images of 10,575 real identities
