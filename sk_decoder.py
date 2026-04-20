import torch
import torch.nn as nn
import torch.nn.functional as f

class BF_batchNorm(nn.Module):
    """
    Custom Batch Normalization from the reference UNet implementation.
    Uses a running standard deviation with learnable per-channel scale (gammas).
    
    Reference: https://github.com/lingqiz/optimal-measurement/tree/unet
    """
    def __init__(self, num_kernels: int):
        super().__init__()
        self.register_buffer("running_sd", torch.ones(1, num_kernels, 1, 1))
        # Initializing gammas with a small random value, clamped.
        g = (torch.randn((1, num_kernels, 1, 1)) * (2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        if self.training:
            # Calculate variance across batch, height, and width
            sd_x = torch.sqrt(x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                # Update running standard deviation with 0.1 momentum
                self.running_sd.copy_(0.9 * self.running_sd.data + 0.1 * sd_x)
            x = x * self.gammas.expand_as(x)
        else:
            # At inference, use the running standard deviation
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)
        return x

class Decoder(nn.Module):
    """
    A UNet-like decoder to reconstruct images from Retinal Ganglion Cell (RGC) responses.
    
    This implementation takes inspiration from the paper:
    'Generalized Compressed Sensing for Image Reconstruction with Diffusion Probabilistic Models'
    (https://openreview.net/forum?id=lmHh4FmPWZ)
    and its implementation: https://github.com/lingqiz/optimal-measurement/tree/unet
    
    Since the Retina model acts as the encoder (mapping frames to cell responses),
    this Decoder module replaces the first half of a standard UNet with a linear projection
    that expands RGC responses into a feature map, followed by a series of 
    upsampling/decoding blocks to reconstruct the original image.
    
    Architecture:
    1. Linear Projection: (total_cells) -> (C * H_bottleneck * W_bottleneck)
    2. Mid-Block: Convolutional layers at the bottleneck resolution.
    3. Decoder Blocks: Sequential stages of ConvTranspose2d (upsampling) and Conv2d blocks.
    4. Output: Final 1-channel grayscale reconstruction.
    """

    def __init__(self, params: dict):
        """
        Initialize the decoder with configuration parameters.
        
        Args:
            params (dict): Dictionary identifying:
                - n_cells: Total number of RGC responses (e.g., n_mosaics * cells_per_mosaic).
                - frame_shape: Final resolution (H, W).
                - num_blocks: Number of upsampling stages (default 3).
                - num_kernels: Base channel count (default 64).
                - kernel_size: Convolution kernel size (default 3).
                - padding: Convolution padding (default 1).
                - bias: Enable bias in convolutional layers (default False).
        """
        super().__init__()
        
        self.params = params
        self.target_h, self.target_w = params["frame_shape"]
        self.n_cells = params["n_cells"]
        
        self.num_blocks = params.get("num_blocks", 3)
        self.num_kernels = params.get("num_kernels", 64)
        self.kernel_size = params.get("kernel_size", 3)
        self.padding = params.get("padding", 1)
        self.bias = params.get("bias", False)
        
        # Calculate bottleneck dimensions (2^num_blocks reduction)
        self.h0 = self.target_h // (2 ** self.num_blocks)
        self.w0 = self.target_w // (2 ** self.num_blocks)
        
        # Channel count at the bottleneck
        self.bottleneck_ch = self.num_kernels * (2 ** self.num_blocks)
        
        # 1. Linear projection to feature space
        self.w = nn.Linear(self.n_cells, self.bottleneck_ch * self.h0 * self.w0)
        
        # 2. Mid Block
        self.mid_block = nn.Sequential(
            nn.Conv2d(self.bottleneck_ch, self.bottleneck_ch, self.kernel_size, padding=self.padding, bias=self.bias),
            BF_batchNorm(self.bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_ch, self.bottleneck_ch, self.kernel_size, padding=self.padding, bias=self.bias)
        )
        
        # 3. Upsampling and Decoder stages
        self.upsample = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        
        for b in range(self.num_blocks - 1, -1, -1):
            in_ch = self.num_kernels * (2 ** (b + 1))
            out_ch = self.num_kernels * (2 ** b)
            
            # Upsample block
            self.upsample[str(b)] = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=self.bias)
            
            # Conv block after upsampling
            if b == 0:
                # Final block to output image size
                self.decoder[str(b)] = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
                    BF_batchNorm(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, 1, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
                )
            else:
                self.decoder[str(b)] = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
                    BF_batchNorm(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
                    BF_batchNorm(out_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image reconstruction.
        
        Supports inputs from the Retina model:
        - (n_mosaics, n_cells, T): Concatenates mosaics across cell dimension.
        - (n_cells, T): Single mosaic input.
        - (batch, n_cells): Direct feature input.
        
        Returns:
            torch.Tensor: Reconstructed frames of shape (T, 1, H, W).
        """
        # --- Handle Retina output shapes ---
        if x.dim() == 4:
            # (batch, n_mosaics, n_cells, temp) -> (batch, n_mosaics * n_cells * temp)
            batch_size, n_mosaics, n_cells, temp = x.shape
            x = x.reshape(batch_size, -1)
        elif x.dim() == 3:
            # (n_mosaics, n_cells, T) -> (T, n_mosaics * n_cells)
            n_mosaics, n_cells, T = x.shape
            x = x.permute(2, 0, 1).contiguous().view(T, n_mosaics * n_cells)
        elif x.dim() == 2:
            # If shape is (n_cells, T), transpose to (T, n_cells)
            # Assumption: n_cells > T for typical use cases, or check if likely flipped
            if x.shape[1] > x.shape[0] and x.shape[0] == self.n_cells:
                # Likely (n_cells, batch)
                 pass # batch=T already if we treat it as (batch, n_cells)
            elif x.shape[0] == self.n_cells and x.shape[1] < x.shape[0]:
                x = x.T # Convert from (n_cells, batch) to (batch, n_cells)
        
        batch_size = x.shape[0]
        
        # 1. Project to feature map
        z = self.w(x)
        z = z.view(batch_size, self.bottleneck_ch, self.h0, self.w0)
        
        # 2. Mid block
        z = self.mid_block(z)
        
        # 3. Upsample and Decode
        for b in range(self.num_blocks - 1, -1, -1):
            z = self.upsample[str(b)](z)
            z = self.decoder[str(b)](z)
            
        # Final safety: resize to target if output is slightly off due to rounding
        if z.shape[2:] != (self.target_h, self.target_w):
            z = f.interpolate(z, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
            
        return z

if __name__ == "__main__":
    # Internal test/demo
    test_params = {
        "n_cells": 1000,
        "frame_shape": (256, 256),
        "num_blocks": 3,
        "num_kernels": 32
    }
    model = Decoder(test_params)
    print(f"Decoder initialized. Bottleneck: {model.h0}x{model.w0}")
    
    # Simulate Retina output: (1 mosaic, 1000 cells, 5 frames)
    dummy_input = torch.randn(1, 1000, 5)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Simulate pre-batched input: (10 batch, 1000 cells)
    dummy_batch = torch.randn(10, 1000)
    output_batch = model(dummy_batch)
    print(f"Batch input shape: {dummy_batch.shape}")
    print(f"Batch output shape: {output_batch.shape}")