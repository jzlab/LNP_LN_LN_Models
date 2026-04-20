import os
import argparse
import torch
import numpy as np
from sk_utils import Utils as u
from sk_models import Retina

def load_and_crop_video(video_path, x_start, y_start, target_h, target_w):
    """
    Loads a video and applies a spatial crop.
    """
    print(f"Loading video from {video_path}...")
    full_video, fps = u.read_video(video_path)
    T, H_full, W_full = full_video.shape
    print(f"Full video loaded: {T} frames, {H_full}x{W_full} resolution, {fps} FPS.")

    x_end, y_end = x_start + target_w, y_start + target_h

    if x_end > W_full or y_end > H_full:
        raise ValueError(
            f"Requested crop ({target_w}x{target_h}) at ({x_start}, {y_start}) "
            f"exceeds video dimensions ({W_full}x{H_full})."
        )

    print(f"Cropping video to {target_h}x{target_w} starting at ({x_start}, {y_start})...")
    video_frames = full_video[:, y_start:y_end, x_start:x_end]
    
    return video_frames, fps, (H_full, W_full)

def setup_retina(params, fps, cell_minibatch=None, temp_batch=None, device="cpu"):
    """
    Initializes the Retina model with specific parameters and moves it to device.
    """
    print("Initializing Retina model...")
    video_params = params.get("video_parameters", {})
    # Update frame_rate to match the actual video FPS
    video_params["frame_rate"] = fps
    
    retina = Retina(
        params, 
        video_params, 
        cell_minibatch_size=cell_minibatch, 
        temporal_batch_size=temp_batch
    )
    retina.to(device)
    retina.eval()
    return retina

def get_activations(video_frames, retina, device="cpu"):
    """
    Runs the forward pass of the retina model on a video.
    """
    print("Running forward pass (vectorized)...")
    video_tensor = torch.from_numpy(video_frames).to(device).float()
    
    with torch.no_grad():
        # retina() returns (linear_responses, firing_rates)
        _, firing_rates = retina(video_tensor)
        
    return firing_rates

def save_activations(firing_rates, output_path, video_path, fps, crop_shape, x_start, y_start, mosaic_names):
    """
    Saves the generated activations and metadata to a file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving activations to {output_path}...")
    output_data = {
        "firing_rates": firing_rates.cpu(),
        "video_info": {
            "path": video_path,
            "fps": fps,
            "shape": crop_shape,
            "x": x_start,
            "y": y_start
        },
        "mosaic_names": mosaic_names
    }
    torch.save(output_data, output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Generate RGC activations from a video using the Retina model.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to the params.yaml file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output activations (e.g., activations.pt).")
    parser.add_argument("--x", type=int, default=0, help="Top-left X coordinate (column) for the spatial crop.")
    parser.add_argument("--y", type=int, default=0, help="Top-left Y coordinate (row) for the spatial crop.")
    parser.add_argument("--cell-minibatch", type=int, default=500, help="Number of cells to process in a single vectorized batch.")
    parser.add_argument("--temp-batch", type=int, default=50, help="Number of temporal windows to process in a single vectorized batch.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")

    args = parser.parse_args()

    # 1. Load parameters
    params = u.read_params(args.params)
    video_params = params.get("video_parameters", {})
    target_h, target_w = video_params.get("frame_shape", [256, 256])
    
    # 2. Load and crop video
    video_frames, fps, _ = load_and_crop_video(args.video, args.x, args.y, target_h, target_w)

    # 3. Setup Retina model
    retina = setup_retina(params, fps, args.cell_minibatch, args.temp_batch, args.device)

    # 4. Generate activations
    firing_rates = get_activations(video_frames, retina, args.device)

    # 5. Save output
    mosaic_names = [m.m_params["cell_type"] for m in retina.mosaics]
    save_activations(firing_rates, args.output, args.video, fps, (len(video_frames), target_h, target_w), args.x, args.y, mosaic_names)

if __name__ == "__main__":
    main()
