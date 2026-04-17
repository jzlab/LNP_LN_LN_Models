import torch
import torch.nn as nn
import torch.nn.functional as f

class Decoder(nn.Module):
    """
    A convolutional decoder to reconstruct the image from the RGC responses.
    Takes inspiration from the following paper:

    Generalized Compressed Sensing for Image Reconstruction with Diffusion Probabilistic Models
    https://openreview.net/forum?id=lmHh4FmPWZ

    Since we use the RGC Mosaic to encode the images, we chop off the first half of the model
    and use the second half as a decoder to reconstruct the image.
    """

    def __init__(self, params):
        super().__init__()
        
        self.params = params
        self.w = nn.Linear(params["n_cells"], params["frame_shape"][0] * params["frame_shape"][1])
    
    def forward(self, x):
        return self.w(x)