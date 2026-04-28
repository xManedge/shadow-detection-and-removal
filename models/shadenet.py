import torch
import torch.nn as nn
from .modules import Down, Attention_block, Decoder_Up, DoubleConv, OutConv, SelfAttention


class ShadeNet(nn.Module):
    """
    ShadeNet: A dual-head Attention U-Net for simultaneous segmentation
    and image reconstruction.

    ShadeNet extends the classic U-Net with two types of attention:
    - SelfAttention at the bottleneck (level 4) to capture global
      long-range dependencies before the decoder begins.
    - Attention_block (additive attention gates, Oktay et al. 2018)
      at levels 1-3 to filter encoder skip features using decoder context,
      suppressing irrelevant spatial activations before concatenation.

    The decoder produces two independent outputs from the same shared
    feature representation:
    - A segmentation mask   (1 channel,  logits → apply sigmoid)
    - A reconstruction image(3 channels, logits → apply sigmoid/tanh)

    Architecture overview (input: (B, 3, 640, 480))
    -------------------------------------------------
    Encoder
        encoder1  : DoubleConv  (B,   3, 640, 480) → (B,  64, 640, 480)
        encoder2  : Down        (B,  64, 640, 480) → (B, 128, 320, 240)
        encoder3  : Down        (B, 128, 320, 240) → (B, 256, 160, 120)
        encoder4  : Down        (B, 256, 160, 120) → (B, 512,  80,  60)
        bottleneck: Down        (B, 512,  80,  60) → (B, 512,  40,  30)

    Decoder (with skip connections and attention)
        Level 4:
            decoder4  : Decoder_Up  (B, 512,  40,  30) → (B, 512,  80,  60)
            attention4: SelfAttention on x4             → (B, 512,  80,  60)
            cat + conv: (B, 1024, 80, 60)               → (B, 512,  80,  60)
        Level 3:
            decoder3  : Decoder_Up  (B, 512,  80,  60) → (B, 256, 160, 120)
            attention3: Attention_block(dec_out, x3)    → (B, 256, 160, 120)
            cat + conv: (B,  512, 160, 120)             → (B, 256, 160, 120)
        Level 2:
            decoder2  : Decoder_Up  (B, 256, 160, 120) → (B, 128, 320, 240)
            attention2: Attention_block(dec_out, x2)    → (B, 128, 320, 240)
            cat + conv: (B,  256, 320, 240)             → (B, 128, 320, 240)
        Level 1:
            decoder1  : Decoder_Up  (B, 128, 320, 240) → (B,  64, 640, 480)
            attention1: Attention_block(dec_out, x1)    → (B,  64, 640, 480)
            cat + conv: (B,  128, 640, 480)             → (B,  64, 640, 480)

    Output heads (both at full resolution 640×480)
        maskOutConv          : (B,  64, 640, 480) → (B,   1, 640, 480)  segmentation logits
        reconstructionOutConv: (B,  64, 640, 480) → (B,   3, 640, 480)  reconstruction logits

    Parameters
    ----------
    mid_layers : list of int, length 5
        Channel widths at each stage: [enc1, enc2, enc3, enc4, bottleneck].
        Expected: [64, 128, 256, 512, 512].
    n_classes : int, optional
        Number of output channels for the segmentation head. Default: 1
        (binary segmentation). For multiclass, set this to the number of
        classes and use CrossEntropyLoss instead of BCEWithLogitsLoss.

    Raises
    ------
    AssertionError
        If ``len(mid_layers) != 5``.

    Notes
    -----
    - Input spatial dimensions (H, W) must be divisible by 2⁴ = 16 due to
      four downsampling stages. 640×480 satisfies this (640/16=40, 480/16=30).
    - Both output heads share the same decoder features — the two tasks
      act as implicit regularisers for each other during training.
    - Segmentation output: pass logits to nn.BCEWithLogitsLoss (binary) or
      nn.CrossEntropyLoss (multiclass). Apply torch.sigmoid at inference.
    - Reconstruction output: pass logits to nn.MSELoss or nn.L1Loss.
      Apply torch.sigmoid at inference if target images are in [0, 1].

    Example
    -------
    >>> model = ShadeNet(mid_layers=[64, 128, 256, 512, 512], n_classes=1)
    >>> x = torch.randn(2, 3, 640, 480)
    >>> mask_logits, reconstruction = model(x)
    >>> mask_logits.shape
    torch.Size([2, 1, 640, 480])
    >>> reconstruction.shape
    torch.Size([2, 3, 640, 480])
    """

    def __init__(self, mid_layers, n_classes=1):
        """
        Initialise ShadeNet encoder, attention, decoder, and output heads.

        Parameters
        ----------
        mid_layers : list of int, length 5
            Channel widths at each network stage in order:
                [encoder1, encoder2, encoder3, encoder4, bottleneck]
            With mid_layers = [64, 128, 256, 512, 512] and a (B, 3, 640, 480)
            input, spatial dimensions at each stage are:
                encoder1   : (B,  64, 640, 480)
                encoder2   : (B, 128, 320, 240)
                encoder3   : (B, 256, 160, 120)
                encoder4   : (B, 512,  80,  60)
                bottleneck : (B, 512,  40,  30)
        n_classes : int, optional
            Number of segmentation output channels. Default: 1.

        Raises
        ------
        AssertionError
            If mid_layers does not have exactly 5 elements.
        """
        super().__init__()

        assert len(mid_layers) == 5, "ShadeNet only has 4 encoders; len(mid_layers) == 5"

        # ── Encoder layers ─────────────────────────────────────────────────
        # encoder1: no spatial downsampling, just feature extraction
        # (B, 3, 640, 480) → (B, 64, 640, 480)
        self.encoder1 = DoubleConv(3, mid_layers[0])

        # encoder2-4 + bottleneck: DoubleConv then MaxPool2d (stride 2)
        # (B,  64, 640, 480) → (B, 128, 320, 240)
        self.encoder2 = Down(mid_layers[0], mid_layers[1])
        # (B, 128, 320, 240) → (B, 256, 160, 120)
        self.encoder3 = Down(mid_layers[1], mid_layers[2])
        # (B, 256, 160, 120) → (B, 512,  80,  60)
        self.encoder4 = Down(mid_layers[2], mid_layers[3])
        # (B, 512,  80,  60) → (B, 512,  40,  30)
        self.bottleneck = Down(mid_layers[3], mid_layers[4])

        # ── Decoder level 4 (bottleneck → encoder4 resolution) ────────────
        # Decoder_Up : (B, 512, 40, 30) → (B, 512, 80, 60)
        self.decoder4 = Decoder_Up(mid_layers[4], mid_layers[3])
        self.attention4 = SelfAttention(mid_layers[3])
        self.up_conv4 = DoubleConv(2 * mid_layers[3], mid_layers[3])

        # ── Decoder level 3 ────────────────────────────────────────────────
        # Decoder_Up : (B, 512,  80,  60) → (B, 256, 160, 120)
        self.decoder3 = Decoder_Up(mid_layers[3], mid_layers[2])
        self.attention3 = Attention_block(mid_layers[2], mid_layers[2])
        self.up_conv3 = DoubleConv(2 * mid_layers[2], mid_layers[2])

        # ── Decoder level 2 ────────────────────────────────────────────────
        # Decoder_Up : (B, 256, 160, 120) → (B, 128, 320, 240)
        self.decoder2 = Decoder_Up(mid_layers[2], mid_layers[1])
        self.attention2 = Attention_block(mid_layers[1], mid_layers[1])
        self.up_conv2 = DoubleConv(2 * mid_layers[1], mid_layers[1])

        # ── Decoder level 1 ────────────────────────────────────────────────
        # Decoder_Up : (B, 128, 320, 240) → (B, 64, 640, 480)
        self.decoder1 = Decoder_Up(mid_layers[1], mid_layers[0])
        self.attention1 = Attention_block(mid_layers[0], mid_layers[0])
        self.up_conv1 = DoubleConv(2 * mid_layers[0], mid_layers[0])

        # ── Output heads ───────────────────────────────────────────────────
        # Segmentation: (B, 64, 640, 480) → (B, n_classes, 640, 480)
        self.maskOutConv = OutConv(mid_layers[0], n_classes)
        # Reconstruction: (B, 64, 640, 480) → (B, 3, 640, 480)
        self.reconstructionOutConv = OutConv(mid_layers[0], 3)

    def forward(self, x):
        """
        Forward pass of ShadeNet.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 3, H, W).
            H and W must each be divisible by 16 (four 2× downsampling stages).
            Expected: (B, 3, 640, 480).

        Returns
        -------
        logits : torch.Tensor
            Segmentation logits of shape (B, n_classes, H, W).
            Pass to BCEWithLogitsLoss during training; apply sigmoid at inference.
        reconstructed : torch.Tensor
            Reconstruction logits of shape (B, 3, H, W).
            Pass to MSELoss / L1Loss during training; apply sigmoid at inference.

        Notes
        -----
        Encoder skip tensors x1–x4 are held in memory until their respective
        decoder level consumes them. Peak memory therefore includes activations
        at all five resolutions simultaneously.
        """

        # ── Encoder (feature extraction + progressive downsampling) ────────
        x1 = self.encoder1(x)          # (B,  64, 640, 480)
        x2 = self.encoder2(x1)         # (B, 128, 320, 240)
        x3 = self.encoder3(x2)         # (B, 256, 160, 120)
        x4 = self.encoder4(x3)         # (B, 512,  80,  60)
        x5 = self.bottleneck(x4)       # (B, 512,  40,  30)

        # ── Decoder level 4 ────────────────────────────────────────────────
        # Upsample bottleneck, gate encoder4 skip with self-attention,
        # concatenate and refine.
        dec_out = self.decoder4(x5)             # (B, 512,  80,  60)
        v4 = self.attention4(x4)                # (B, 512,  80,  60)
        dec_out = torch.cat((dec_out, v4), dim=1)  # (B, 1024, 80,  60)
        dec_out = self.up_conv4(dec_out)        # (B, 512,  80,  60)

        # ── Decoder level 3 ────────────────────────────────────────────────
        # Upsample, gate encoder3 skip using decoder signal, concatenate.
        dec_out = self.decoder3(dec_out)           # (B, 256, 160, 120)
        v3 = self.attention3(dec_out, x3)          # (B, 256, 160, 120)
        dec_out = torch.cat((dec_out, v3), dim=1)  # (B, 512, 160, 120)
        dec_out = self.up_conv3(dec_out)           # (B, 256, 160, 120)

        # ── Decoder level 2 ────────────────────────────────────────────────
        dec_out = self.decoder2(dec_out)           # (B, 128, 320, 240)
        v2 = self.attention2(dec_out, x2)          # (B, 128, 320, 240)
        dec_out = torch.cat((dec_out, v2), dim=1)  # (B, 256, 320, 240)
        dec_out = self.up_conv2(dec_out)           # (B, 128, 320, 240)

        # ── Decoder level 1 ────────────────────────────────────────────────
        dec_out = self.decoder1(dec_out)           # (B,  64, 640, 480)
        v1 = self.attention1(dec_out, x1)          # (B,  64, 640, 480)
        dec_out = torch.cat((dec_out, v1), dim=1)  # (B, 128, 640, 480)
        dec_out = self.up_conv1(dec_out)           # (B,  64, 640, 480)

        # ── Output heads ───────────────────────────────────────────────────
        logits        = self.maskOutConv(dec_out)           # (B,   1, 640, 480)
        reconstructed = self.reconstructionOutConv(dec_out) # (B,   3, 640, 480)

        return logits, reconstructed