# shadow-detection-and-removal
# Shadow Detection and Removal using U-Net + Transformer Attention

This project focuses on shadow detection (segmentation) and shadow removal (image reconstruction) using a hybrid deep learning architecture combining a U-Net backbone with transformer-style attention modules.

The model is trained on the ISTD (Image Shadow Triplets Dataset).

---

# Problem Statement

Shadows in images degrade visual quality and can negatively affect downstream computer vision tasks such as object detection and scene understanding.

This project addresses two tasks:
1. Shadow detection (segmentation)
2. Shadow removal (image-to-image reconstruction)

---

# Dataset: ISTD (Image Shadow Triplets Dataset)

The ISTD dataset contains paired training samples:

- Shadow image (input)
- Shadow-free image (ground truth)
- Shadow mask (binary annotation)

Key characteristics:
- Real-world outdoor scenes
- Pixel-level supervision for shadows
- Standard benchmark dataset for shadow detection and removal

Why ISTD:
- Supports both segmentation and reconstruction tasks
- Widely used in shadow removal research
- Provides aligned supervision for multi-task learning

---

# Model Architecture

The model uses a multi-task design:

Input Image  
→ U-Net Encoder (CNN backbone)  
→ Transformer Attention Block (global context modeling)  
→ Shared feature representation  
→ Two task-specific heads:

1. Shadow Segmentation Head  
   - Outputs binary shadow mask

2. Shadow Removal Decoder  
   - Outputs shadow-free RGB image

---


# Model Architecture

The model is a **multi-task U-Net with a shared encoder and two decoders**.

##  Overall Pipeline


Input Image (RGB)
↓
Shared Encoder (CNN + Attention)
↓
Shared Feature Map
↓
────────────────────────────
│ │
↓ ↓
Segmentation Decoder Removal Decoder
↓ ↓
Shadow Mask Shadow-Free Image


---



# Tasks

## Shadow Segmentation
Predict pixel-wise shadow mask.

Output:
- Binary mask (shadow / non-shadow)

## Shadow Removal
Generate a shadow-free image from input.

Output:
- RGB reconstructed image

---

# Loss Functions

## Dice Loss (Segmentation)

Dice loss is used to handle class imbalance:

\[
\mathcal{L}_{dice} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}
\]

Dice loss is appropriate for this task because:
- shadow pixels are sparse compared to background
- it improves boundary quality over BCE alone

---

## Intersection over Union (IoU)

Used as evaluation metric:

\[
IoU = \frac{Prediction \cap GroundTruth}{Prediction \cup GroundTruth}
\]

mIoU is a standard metric for segmentation quality.

---

## Reconstruction Loss (Shadow Removal)

- L1 loss is used as the primary reconstruction loss
- SSIM loss can optionally improve perceptual quality

---

# Training Strategy

Recommended training procedure:

1. Train segmentation branch independently first
2. Add shadow removal branch
3. Joint training with weighted loss balancing

---

# Challenges

- Shadow boundaries are ambiguous
- Dark objects can be confused with shadows
- Multi-task training can cause gradient conflict
- Transformer blocks increase computational cost

---

# Future Improvements

- Replace encoder with EfficientNet or ResNet variants
- Experiment with Swin Transformer attention
- Add perceptual loss for reconstruction quality
- Improve loss balancing for multi-task stability

---

# Status

Code in progress