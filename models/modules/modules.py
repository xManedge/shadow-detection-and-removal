# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A two-layer convolutional block used throughout U-Net-style architectures.

    This block applies the following sequence twice:
        Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU

    Design intent:
    - The 3x3 kernels with padding=1 preserve spatial resolution (H, W stay the same).
    - Batch normalization stabilizes training and allows for higher learning rates.
    - In-place ReLU saves memory while keeping semantics identical.

    Parameters
    ----------
    in_channel : int
        Number of input feature channels (C_in) in the incoming tensor.
    out_channel : int
        Number of output feature channels (C_out) produced by the block.
    mid_channels : Optional[int]
        Number of channels used by the first conv. If None/falsey, defaults to `out_channel`.
        Setting this explicitly can create a bottleneck or expansion between the two convs.
    bias : bool
        Whether to include bias terms in the Conv2d layers. Often set to False when using
        BatchNorm, since BatchNorm offsets can absorb the bias.

    Input shape
    -----------
    (N, in_channel, H, W)

    Output shape
    ------------
    (N, out_channel, H, W)

    Notes
    -----
    - This block is typically used both in the encoder (downsampling pathway) and decoder
      (upsampling pathway) of U-Net to refine features while maintaining spatial size.
    - Using `mid_channels != out_channel` enables a light "compression/expansion" between
      the two convolutions, which can be useful for parameter control.
    """

    def __init__(self, in_channel, out_channel, mid_channels=None, bias=False):
        super().__init__()

        # If an intermediate channel count isn't provided, match the final width.
        if not mid_channels:
            mid_channels = out_channel

        # Two consecutive Conv-BN-ReLU blocks with 3x3 kernels and padding=1 to keep H, W unchanged.
        self.doubleconv = nn.Sequential(
            # First 3x3 convolution: C_in -> mid_channels
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,          # keep spatial dimensions (same-conv for 3x3)
                bias=bias
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Second 3x3 convolution: mid_channels -> C_out
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,          # keep spatial dimensions (same-conv for 3x3)
                bias=bias
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channel, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channel, H, W), after two Conv-BN-ReLU stages.
        """
        # Apply the stacked Conv-BN-ReLU layers.
        return self.doubleconv(x)


class Down(nn.Module):
    """
    Downsampling block (variant): DoubleConv → MaxPool2d

    This encoder-stage block first refines features at the current resolution
    using a DoubleConv (Conv-BN-ReLU ×2) and then downsamples spatially with a
    2×2 max pooling operation. The ordering:
        1) Feature extraction/refinement at full resolution
        2) Spatial decimation by a factor of 2 (H/2, W/2)

    Rationale
    ---------
    - Applying convolutions before pooling lets the network extract richer,
      high-frequency features while full spatial detail is still available.
    - MaxPooling then reduces resolution and increases the effective receptive
      field for subsequent stages.

    Parameters
    ----------
    in_channels : int
        Number of input channels (C_in).
    out_channels : int
        Number of output channels after the DoubleConv (C_out).

    Input shape
    -----------
    (N, in_channels, H, W)

    Output shape
    ------------
    (N, out_channels, H/2, W/2)

    Notes
    -----
    - This variant differs from the classic U-Net ordering (MaxPool → DoubleConv).
      Make sure skip connections and decoder expectations align with this design.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Keep your original ordering: first refine at full resolution, then pool.
        self.conv_pool = nn.Sequential(
            # 1) Two 3×3 Conv-BN-ReLU layers (preserve H, W via padding=1)
            DoubleConv(in_channels, out_channels),
            # 2) Downsample by 2× in each spatial dimension
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, out_channels, H/2, W/2) after DoubleConv then MaxPool.
        """
        return self.conv_pool(x)
    
    
class SelfAttention(nn.Module): 
    """
    Optimized Self-Attention block for convolutional feature maps using Flash Attention.

    Purpose
    -------
    Computes relationships between spatial positions in a feature map to model long-range
    dependencies. This optimized version uses PyTorch's `scaled_dot_product_attention`
    to significantly reduce memory usage and improve speed.

    Architecture Context
    --------------------
    - Uses 1x1 convolutions to project inputs into Query, Key, and Value spaces.
    - Leverages **Flash Attention** (via `F.scaled_dot_product_attention`) to avoid 
      materializing the massive N x N attention matrix.
    - Includes a residual connection scaled by a learnable parameter gamma.

    Parameters
    ----------
    in_dim : int
        Number of input feature channels (C).

    Input shape
    -----------
    x : torch.Tensor
        Feature map of shape (N, C, H, W).

    Output shape
    ------------
    torch.Tensor
        Refined feature map of shape (N, C, H, W) after applying self-attention.

    Notes
    -----
    - **Optimization:** Replaced manual matrix multiplication (O(N^2) memory) with
      Flash Attention (O(N) memory). This prevents OOM errors on large maps (e.g., 128x128).
    - **Scaling:** Automatically applies the 1/sqrt(dim) scaling factor for numerical stability.
    - The learnable parameter `gamma` starts at 0, allowing the network to initially 
      act as an identity mapping and gradually learn the attention weights.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.channel = in_dim
        
        # Linear projections for Query, Key, Value (via 1x1 convolutions)
        # We reduce channels by 8 for Query and Key to reduce computational cost
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Learnable scaling factor for residual connection
        # Initialized to 0 to start as an identity mapping
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of the optimized self-attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        out : torch.Tensor
            Tensor of shape (N, C, H, W) after applying self-attention and residual connection.
        """
        batch_size, C, height, width = x.size()
        N = height * width  # Sequence length (spatial pixels)

        # 1. Projections
        # Reshape to (Batch, SeqLen, Dim) for Flash Attention
        # Q: (B, C//8, H, W) -> (B, C//8, N) -> (B, N, C//8)
        q = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        # K: (B, C//8, H, W) -> (B, C//8, N) -> (B, N, C//8)
        k = self.key_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        # V: (B, C, H, W)    -> (B, C, N)    -> (B, N, C)
        v = self.value_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        
        # 2. Optimized Attention (Flash / Memory Efficient)
        # Automatically handles the scaling factor: 1 / sqrt(dim)
        # Uses efficient kernels to avoid O(N^2) memory cost
        out = F.scaled_dot_product_attention(q, k, v)
        
        # 3. Reshape back to feature map dimensions
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out = out.permute(0, 2, 1).view(batch_size, C, height, width)
        
        # 4. Residual connection with learnable scaling
        out = self.gamma * out + x
        
        return out



class Attention_block(nn.Module): 
    """
    Attention gate for U-Net-style skip connections (encoder "gates" referenced by decoder features).

    Purpose
    -------
    Filters encoder skip features `enc` using a decoder-driven attention signal derived from `dec`.
    This lets the network suppress irrelevant encoder activations and pass only context-relevant
    information to the decoder, improving focus around target structures.

    Mechanism
    ---------
    1) Project decoder features (gate) and encoder features into a shared intermediate space:
         - W_gate(dec) : Conv1x1 + BN  → shape (N, I, H, W)
         - W_encoder(enc): Conv1x1 + BN → shape (N, I, H, W)
    2) Combine and activate: ReLU( W_gate(dec) + W_encoder(enc) )
    3) Collapse to a single-channel attention map with a sigmoid:
         - psi(·): Conv1x1 → (N, 1, H, W) → Sigmoid → attention mask in [0,1]
    4) Modulate encoder features: enc * psi (broadcast over channels)

    Parameters
    ----------
    encoder_channels : int
        Number of channels in the encoder skip tensor `enc`.
    decoder_channels : int
        Number of channels in the decoder tensor `dec` that provides the gating context.
    intermediate_channels : Optional[int]
        Channel width of the shared intermediate space. If not provided, defaults to
        `decoder_channels // 2`.

    Notes
    -----
    - Uses lightweight 1×1 convolutions to align channels before computing attention,
      keeping computation and memory overhead modest.
    - The attention mask is spatial (per-pixel) and broadcast across encoder channels.
    - Typical usage: `att_enc = Attention_block(...)(enc, dec); concat = torch.cat([att_enc, dec], dim=1)`.
    """
    def __init__(self, encoder_channels, decoder_channels, intermediate_channels = None):
        super().__init__()
        if not intermediate_channels: 
            intermediate_channels = decoder_channels // 2
        
        # Project/gate the decoder features into the shared intermediate space.
        self.W_gate = nn.Sequential(
            nn.Conv2d(in_channels= decoder_channels, out_channels= intermediate_channels, kernel_size= 1, stride=1, padding= 0),
            nn.BatchNorm2d(intermediate_channels),
        )
        
        # Project the encoder (skip) features into the same intermediate space.
        self.W_encoder = nn.Sequential(
            nn.Conv2d(in_channels= encoder_channels, out_channels= intermediate_channels, kernel_size= 1, stride= 1, padding= 0),
            nn.BatchNorm2d(intermediate_channels),
        )
        
        # Produce a 1-channel spatial attention map ψ ∈ [0,1] via sigmoid.
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels= intermediate_channels, out_channels= 1, kernel_size= 1, stride = 1, padding= 0),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, enc, dec):
        """
        Forward pass of the attention gate.

        Parameters
        ----------
        enc : torch.Tensor
            Encoder skip connection features of shape (N, encoder_channels, H, W).
        dec : torch.Tensor
            Decoder features (gating signal) of shape (N, decoder_channels, H, W).

        Returns
        -------
        torch.Tensor
            Attention-weighted encoder features of shape (N, encoder_channels, H, W).
            These are element-wise modulated by the spatial attention mask derived
            from both encoder and decoder signals.
        """
        # Decoder-driven gating signal and encoder projection (both to intermediate width).
        dec_conv = self.W_gate(dec)
        enc_conv = self.W_encoder(enc)
        
        # Combine in the shared space, then activate.
        psi = self.relu(dec_conv + enc_conv)
        # Collapse to a spatial attention mask (N,1,H,W) in [0,1].
        psi = self.psi(psi)
        
        # Reweight encoder features channel-wise by broadcasting the spatial mask.
        return enc * psi


class Decoder_Up(nn.Module): 
    """
    Upsampling block for Attention U-Net decoder and the base U-Net.

    Purpose
    -------
    This block upsamples decoder features to the spatial resolution of the
    corresponding encoder stage. In Attention U-Net, the upsampled decoder
    features are concatenated with encoder features that have first been
    filtered by an attention gate.

    Architecture Context
    --------------------
    - Replaces the original ConvTranspose2d with **Bilinear Upsampling**.
    - This eliminates "checkerboard artifacts" often caused by transposed convolutions
      and reduces the parameter count, making the model lighter and faster.
    - In the decoder path:
        1) Decoder features are upsampled by a factor of 2 using bilinear interpolation.
        2) The upsampled features are refined with a DoubleConv block.

    Parameters
    ----------
    in_channel : int
        Number of channels in the incoming decoder feature map.
    out_channel : int
        Number of channels in the output after upsampling and DoubleConv refinement.

    Input shape
    -----------
    x : torch.Tensor
        Decoder feature map of shape (N, in_channel, H, W).

    Output shape
    ------------
    torch.Tensor
        Upsampled and refined decoder feature map of shape (N, out_channel, 2H, 2W),
        aligned spatially to the corresponding encoder skip resolution.

    Notes
    -----
    - **Optimization:** Uses parameter-free bilinear upsampling instead of ConvTranspose2d.
      This is generally faster and prevents artifact generation in segmentation masks.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        # Optimization: Switched to Bilinear Upsampling + DoubleConv
        # This replaces the heavier ConvTranspose2d layer.
        self.up = nn.Sequential(
            # Step 1: Upsample by factor of 2 with bilinear interpolation
            # align_corners=True preserves pixel alignment critical for segmentation
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Step 2: Refine the upsampled features with DoubleConv
            # Note: Input channels to DoubleConv remain 'in_channel' because 
            # nn.Upsample does not change channel depth.
            DoubleConv(in_channel, out_channel)
        )
    
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Decoder feature map of shape (N, in_channel, H, W).

        Returns
        -------
        torch.Tensor
            Upsampled and refined feature map of shape (N, out_channel, 2H, 2W).
        """
        return self.up(x)
    


class OutConv(nn.Module):
    """
    Final projection layer for U-Net variants.

    Purpose
    -------
    Applies a 1×1 convolution to map the decoder's final feature map to the
    desired number of output channels (e.g., number of classes for segmentation).
    The 1×1 kernel mixes channels without changing spatial resolution.

    Parameters
    ----------
    in_channels : int
        Number of channels in the incoming decoder feature map.
    out_channels : int
        Number of channels in the output. For segmentation, this is typically
        the number of classes (C).

    Input shape
    -----------
    (N, in_channels, H, W)

    Output shape
    ------------
    (N, out_channels, H, W)

    Notes
    -----
    - This layer does not apply activation. For training:
        * Multiclass: pass outputs to `nn.CrossEntropyLoss` (expects raw logits).
        * Binary/multilabel: pass outputs to `nn.BCEWithLogitsLoss` or apply
          a sigmoid at inference.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1×1 conv: channel-wise linear projection, preserves (H, W)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Decoder feature map of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (N, out_channels, H, W).
        """
        return self.conv(x)



