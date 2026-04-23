# LNP_LN_LN_Models

**Forked from:** [https://github.com/NilouGhazavi/LNP_LN_LN_Models](https://github.com/NilouGhazavi/LNP_LN_LN_Models)

PyTorch implementation of Linear-Nonlinear-Poisson (LNP) models for Retinal Ganglion Cells (RGCs), combined with a neural decoder to reconstruct video frames from simulated RGC firing patterns.

## Installation

Create and activate the required Conda environment:

```bash
conda env create -f environment.yml
conda activate lnp_torch
```

## Pipeline Setup and Execution

The pipeline consists of three stages: data generation, neural decoder training, and video reconstruction.

### 1. Generate RGC Activations
Generates simulated RGC activations from input videos and saves them as `.h5` files. Supports processing single videos or batch processing entire directories.

```bash
python sk_generate_activations.py \
    --video lnp_naturalmovies/ \
    --params params.yaml \
    --output lnp_activations_bank \
    --random-crops 25
```

### 2. Train the Decoder
Trains a PyTorch Decoder mapping RGC firing rates back to original video crops. The dataset loader memory-maps the serialized `.h5` files to handle large multi-video structures efficiently.

```bash
python sk_train_decoder.py \
    --activations lnp_activations_bank \
    --params params.yaml \
    --training-params params_training.yaml \
    --output best_decoder.pt \
    --epochs 50 \
    --device cuda
```

### 3. Reconstruct Target Videos
Chops a target video into spatial patches, computes responses, decodes them using the trained model weights, and writes the stitched patches into an output `.mp4`.

```bash
python sk_reconstruct_video.py \
    --video lnp_naturalmovies/Video_2.mp4 \
    --weights best_decoder.pt \
    --output-video Video_2_reconstructed.mp4 \
    --device cuda
```

## Configuration Files

- **`params.yaml`**: Configures the spatial/temporal constraints and convolution parameters for the forward Retina LNP simulation.
- **`params_training.yaml`**: Defines decoder architecture properties and basic training hyperparameters.