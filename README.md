# ShadeNet

ShadeNet is a dual-head Attention U-Net for simultaneous shadow detection and shadow removal. Given an input RGB image, the model jointly predicts a binary shadow mask and a shadow-free reconstruction of the original image.

---

## Table of Contents

- [Architecture](#architecture)
- [Losses](#losses)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Outputs](#outputs)
- [Results](#results)

---

## Architecture

ShadeNet is built on a U-Net backbone with two types of attention and two output heads.

### Encoder

The encoder progressively downsamples the input image through four stages plus a bottleneck. Each stage applies a DoubleConv block (two consecutive Conv2d-BatchNorm-ReLU sequences with 3x3 kernels) followed by MaxPool2d for spatial downsampling.

```
Input  (B,   3, 640, 480)
enc1   (B,  64, 640, 480)   DoubleConv only, no downsampling
enc2   (B, 128, 320, 240)   DoubleConv + MaxPool
enc3   (B, 256, 160, 120)   DoubleConv + MaxPool
enc4   (B, 512,  80,  60)   DoubleConv + MaxPool
bottle (B, 512,  40,  30)   DoubleConv + MaxPool + Dropout2d
```

### Decoder

The decoder upsamples back to the original resolution using bilinear interpolation followed by DoubleConv refinement. Skip connections from the encoder are modulated by attention gates before being concatenated with decoder features.

```
Level 4: Upsample bottle -> (B, 512, 80, 60)
         SelfAttention on enc4 skip
         Concat + DoubleConv -> (B, 512, 80, 60)

Level 3: Upsample -> (B, 256, 160, 120)
         Attention_block(dec, enc3)
         Concat + DoubleConv -> (B, 256, 160, 120)

Level 2: Upsample -> (B, 128, 320, 240)
         Attention_block(dec, enc2)
         Concat + DoubleConv -> (B, 128, 320, 240)

Level 1: Upsample -> (B, 64, 640, 480)
         Attention_block(dec, enc1)
         Concat + DoubleConv -> (B, 64, 640, 480)
```

### Attention Mechanisms

**SelfAttention (bottleneck level):** Applied to the encoder4 skip connection before it enters the decoder. Uses Flash Attention (via `F.scaled_dot_product_attention`) with 1x1 convolution projections for Query (C/8 channels), Key (C/8 channels), and Value (C channels). Includes a learnable residual scaling parameter gamma initialised to zero so the block starts as an identity mapping.

**Attention_block (decoder levels 1-3):** Additive attention gate derived from Oktay et al. 2018. Takes the decoder feature map as the gating signal and the encoder skip as the input. Projects both into a shared intermediate space via 1x1 convolutions, combines them with ReLU, then produces a single-channel spatial attention mask via sigmoid. The mask is broadcast across encoder channels to suppress irrelevant spatial activations before the skip connection is concatenated into the decoder.

### Output Heads

Both heads operate on the final decoder feature map at full input resolution.

- **Segmentation head:** 1x1 Conv2d producing (B, n_classes, H, W) logits. Apply sigmoid at inference for binary shadow masks.
- **Reconstruction head:** 1x1 Conv2d producing (B, 3, H, W) logits. Apply sigmoid at inference for shadow-free RGB images.

---

## Losses

Training minimises a combined loss over both heads simultaneously:

```
total_loss = dice_w * DiceLoss(predicted_mask, mask)
           + mask_w * BCEWithLogitsLoss(predicted_mask, mask)
           + mse_w  * MSELoss(reconstructed, target)
           + perceptual_w * LPIPS(reconstructed, target)
```

### Segmentation Losses

**Binary Dice Loss:** Optimises the overlap between predicted and ground truth mask directly. Robust to class imbalance since it operates on the ratio of intersection to union rather than per-pixel accuracy. Implemented as a custom `BinaryDiceLoss` class without external dependencies.

**BCE with Logits Loss:** Pixel-wise binary cross entropy applied to raw logits via `nn.BCEWithLogitsLoss`. Provides precise boundary-level supervision that complements the region-level Dice loss.

### Reconstruction Losses

**MSE Loss:** Mean squared error between the reconstructed image and the shadow-free target. Provides stable pixel-level supervision with smooth gradients.

**LPIPS (Learned Perceptual Image Patch Similarity):** Perceptual loss computed in VGG16 feature space rather than pixel space. Encourages structural coherence, texture preservation, and edge sharpness in the reconstruction that MSE alone cannot capture. Uses `torchmetrics.image.LearnedPerceptualImagePatchSimilarity` with `net_type='vgg'`.

### Loss Warmup

Segmentation losses are ramped up linearly over the first `warmup_epochs` epochs to allow the reconstruction head to stabilise before full segmentation supervision begins:

```
ramp = min((epoch + 1) / warmup_epochs, 1.0)
loss = dice_w * DiceLoss * ramp + mask_w * BCE * ramp + mse_w * MSE + perceptual_w * LPIPS
```

---

## Dataset

ShadeNet is trained and evaluated on the **ISTD dataset** (Image Shadow Triplets Dataset).

The dataset provides triplets of aligned images:

| Folder | Content |
|--------|---------|
| train_A / test_A | Input RGB images with shadows |
| train_B / test_B | Binary shadow masks (foreground = shadow) |
| train_C / test_C | Shadow-free target RGB images |

- Training set: 1,330 images
- Test set: 540 images

Download the ISTD dataset and place it at the path specified by `traingenerator.root` and `valgenerator.root` in `config.yaml`.

Expected directory structure:

```
Dataset/
└── ISTD_Dataset/
    ├── train/
    │   ├── train_A/
    │   ├── train_B/
    │   └── train_C/
    └── test/
        ├── test_A/
        ├── test_B/
        └── test_C/
```

---

## Project Structure

```
ShadeNet/
├── config/
│   └── config.yaml             # All hyperparameters and paths
├── dataset_generators/
│   └── generator.py            # Generator dataset class
├── models/
│   └── shadenet.py             # ShadeNet model definition
├── modules/
│   └── modules.py              # DoubleConv, Down, SelfAttention,
│                               # Attention_block, Decoder_Up, OutConv
├── utils/
│   ├── train.py                # train_shadenet function
│   └── save.py                 # save_final_model_and_metrics function
├── train.py                    # Training entry point
├── inference.py                # Inference entry point
└── README.md
```

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install torchmetrics[image]
pip install pyyaml
pip install tqdm
pip install Pillow
pip install numpy
```

### Tested Environment

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (training requires GPU; inference can run on CPU)

---

## Configuration

All training and inference parameters are controlled through `config/config.yaml`. No code changes are required between runs.

```yaml
model:
  mid_layers: [64, 128, 256, 512, 512]   # channel widths per stage
  n_classes: 1                            # 1 for binary shadow detection

training:
  epochs: 150
  warmup_epochs: 5                        # epochs to ramp segmentation losses
  lr: 7e-5
  ignore_index: 255
  dice_w: 1.2                             # weight for Dice loss
  mask_w: 0.8                             # weight for BCE loss
  mse_w: 1.0                              # weight for MSE reconstruction loss
  perceptual_w: 0.1                       # weight for LPIPS perceptual loss
  accumulation_steps: 4                   # gradient accumulation steps
  device: cuda
  save_dir: ./models

traingenerator:
  root: ./Dataset/ISTD_Dataset/train
  augment: true
  isTrain: true

valgenerator:
  root: ./Dataset/ISTD_Dataset/test
  augment: false
  isTrain: false

dataloader:
  batch_size: 16
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

transforms:
  img_mean: [0.485, 0.456, 0.406]
  img_std: [0.229, 0.224, 0.225]
  img_resize: [640, 480]
  mask_resize: [640, 480]
```

---

## Training

1. Set dataset paths in `config/config.yaml` under `traingenerator.root` and `valgenerator.root`.
2. Adjust hyperparameters as needed.
3. Run:

```bash
python train.py
```

Training will print per-epoch losses to stdout:

```
Epoch [1/150]
    LR              : 7.00e-05
    Train Loss      : 0.6842
    Train Dice Loss : 0.7486
    Train BCE Loss  : 0.6183
    Train MSE Loss  : 0.0820
    Val Dice Loss   : 0.7750
    Val BCE Loss    : 0.6091
    Val MSE Loss    : 0.0211
```

Saved files after training completes:

| File | Description |
|------|-------------|
| `shadenet.pt` | Model weights (fp32 state dict) |
| `shadenet_fp16.pt` | Model weights converted to fp16 for inference deployment |
| `shadenet_config.pkl` | Saved config file |
| `train_metrics.pkl` | Per-epoch training losses (dice, bce, reconstruction, LPIPS) |
| `val_metrics.pkl` | Per-epoch validation losses |

### Mixed Precision

Training automatically uses fp16 mixed precision via `torch.amp.autocast` and `GradScaler` when `device: cuda` is set. No additional configuration is required. The saved `shadenet.pt` weights remain in fp32 for numerical stability; the separate `shadenet_fp16.pt` is provided for deployment.

### Data Augmentation

The following augmentations are applied to training images only. All spatial transforms are applied identically to the input image, mask, and reconstruction target to preserve alignment. Photometric transforms are applied to the input image and reconstruction target only — never to the mask.

| Transform | Probability | Parameters |
|-----------|-------------|------------|
| Horizontal flip | 0.5 | — |
| Random rotation | 0.5 | angle in [-10, 10] degrees |
| Random affine | 0.5 | translate 5%, scale 95-105% |
| Color jitter | 0.5 | brightness 0.2, contrast 0.2, saturation 0.1, hue 0.02 |
| Gaussian blur | 0.2 | kernel 3, sigma [0.1, 0.5] |

---

## Inference

1. Set the path to a trained model and input images in `config/config.yaml`.
2. Run:

```bash
python inference.py
```

### Outputs

For each input image the model produces two outputs:

**Shadow mask:** A single-channel probability map in [0, 1] after sigmoid activation. Values close to 1 indicate shadow regions. Threshold at 0.5 for a binary mask.

**Shadow-free reconstruction:** A three-channel RGB image in [0, 1] after sigmoid activation representing the input scene with shadows removed.

---

## Results

Best results achieved after 150 epochs with LPIPS and increased segmentation loss weights:

| Metric | Train | Validation |
|--------|-------|------------|
| Dice Loss | 0.030 | 0.118 |
| BCE Loss | 0.019 | 0.157 |
| MSE (Reconstruction) | 0.009 | 0.008 |
| LPIPS | 0.138 | 0.143 |

The reconstruction head generalises well with near-identical train and validation MSE. The segmentation head shows a train-validation gap indicating room for improvement through stronger regularisation.

---

## Notes

- Input images must be RGB. Masks must be grayscale with pixel values 0 (non-shadow) and 1 or 255 (shadow). Pixel value 255 is treated as ignore index during training.
- The model expects input spatial dimensions divisible by 16 due to four downsampling stages. The default configuration uses 640x480. For other resolutions update `img_resize` and `mask_resize` in `config.yaml`.
- If your dataset has significant class imbalance (shadow pixels less than 10% or more than 90% of all pixels), compute `pos_weight` from the mask frequency returned by `dataset.__getstats__()` and pass it to `BCEWithLogitsLoss`.
