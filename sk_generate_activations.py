import os
import argparse
import random
import torch
import numpy as np
import h5py
import json
import scipy.ndimage
from sk_utils import Utils as u
from sk_models import Retina

from tqdm import tqdm

def apply_augmentation(full_video, cx, cy, target_h, target_w, angle, spatial_scale, temporal_scale):
    """
    Applies spatial rotation, spatial scaling, and temporal scaling to a bounding box around (cx, cy).
    """
    T_full, H_full, W_full = full_video.shape
    
    # 1. Determine bounding box for rotation to avoid black corners
    h_src = target_h * spatial_scale
    w_src = target_w * spatial_scale
    D = int(np.ceil(np.sqrt(h_src**2 + w_src**2)))
    
    # Center of the requested target crop
    center_x = cx + target_w / 2.0
    center_y = cy + target_h / 2.0
    
    half_D = int(np.ceil(D / 2.0))
    x_min = int(center_x - half_D)
    x_max = int(center_x + half_D)
    y_min = int(center_y - half_D)
    y_max = int(center_y + half_D)
    
    # Pad if out of bounds
    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - W_full)
    pad_bottom = max(0, y_max - H_full)
    
    valid_x_min = max(0, x_min)
    valid_y_min = max(0, y_min)
    valid_x_max = min(W_full, x_max)
    valid_y_max = min(H_full, y_max)
    
    crop_D = full_video[:, valid_y_min:valid_y_max, valid_x_min:valid_x_max]
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop_D = np.pad(crop_D, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
        
    # 2. Spatial Rotation
    if angle != 0:
        rotated_D = scipy.ndimage.rotate(crop_D, angle, axes=(1, 2), reshape=False, mode='reflect')
    else:
        rotated_D = crop_D
        
    # 3. Center crop to (h_src, w_src)
    ch_src, cw_src = int(np.ceil(h_src)), int(np.ceil(w_src))
    y_start = half_D - ch_src // 2
    x_start = half_D - cw_src // 2
    y_end = y_start + ch_src
    x_end = x_start + cw_src
    
    final_crop = rotated_D[:, y_start:y_end, x_start:x_end]
    
    # 4. PyTorch interpolation for exact sizing and temporal scaling
    vid_tensor = torch.from_numpy(final_crop).unsqueeze(0).float() # (1, T, H, W)
    
    # Spatial resize
    if ch_src != target_h or cw_src != target_w:
        vid_tensor = torch.nn.functional.interpolate(vid_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
    # Temporal scaling
    if temporal_scale != 1.0:
        T_new = int(np.round(T_full * temporal_scale))
        # treat H*W as channels for 1D temporal interpolation
        vid_tensor = vid_tensor.reshape(1, T_full, -1).permute(0, 2, 1) # (1, H*W, T)
        vid_tensor = torch.nn.functional.interpolate(vid_tensor, size=T_new, mode='linear', align_corners=False)
        vid_tensor = vid_tensor.permute(0, 2, 1).reshape(1, T_new, target_h, target_w)
        
    return vid_tensor.squeeze(0).numpy()

def crop_video(full_video, H_full, W_full, x_start, y_start, target_h, target_w):
    """
    Applies a spatial crop to the loaded full video, using zero padding if out of bounds.
    """
    x_end, y_end = x_start + target_w, y_start + target_h

    if x_end <= W_full and y_end <= H_full:
        video_frames = full_video[:, y_start:y_end, x_start:x_end]
    else:
        valid_x_end = min(x_end, W_full)
        valid_y_end = min(y_end, H_full)
        crop = full_video[:, y_start:valid_y_end, x_start:valid_x_end]
        pad_w = target_w - crop.shape[2]
        pad_h = target_h - crop.shape[1]
        video_frames = np.pad(crop, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
    return video_frames

def setup_retina(params, fps, cell_minibatch=None, temp_batch=None, device="cpu"):
    print("Initializing Retina model...")
    video_params = params.get("video_parameters", {})
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
    video_tensor = torch.from_numpy(video_frames).to(device).float()
    
    with torch.no_grad():
        _, firing_rates = retina(video_tensor, pad=False)
        
    return firing_rates

def process_video(video_path, output_path, args, params, target_h, target_w):
    # 2. Load the full video
    print(f"Loading video from {video_path}...")
    full_video, props = u.read_video(video_path)
    T, H_full, W_full = full_video.shape
    fps = props["fps"]
    print(f"Full video loaded: {T} frames, {H_full}x{W_full} resolution, {fps} FPS.")

    # 3. Setup Retina model
    retina = setup_retina(params, fps, args.cell_minibatch, args.temp_batch, args.device)
    mosaic_names = [m.m_params["cell_type"] for m in retina.mosaics]

    # 4. Determine coordinates
    coords = []
    if args.random_crops > 0:
        for _ in range(args.random_crops):
            rx = random.randint(0, max(0, W_full - target_w))
            ry = random.randint(0, max(0, H_full - target_h))
            coords.append((rx, ry))
        print(f"Generated {args.random_crops} random crops: {coords}")
    else:
        x_start = args.x if args.x is not None else 0
        y_start = args.y if args.y is not None else 0
        coords.append((x_start, y_start))

    # 5. Process each crop and its augmentations
    trials = []
    
    for idx, (cx, cy) in tqdm(enumerate(coords), total=len(coords), desc=f"Crops ({os.path.basename(video_path)})"):
        # Base Trial (No Augmentations)
        video_frames = crop_video(full_video, H_full, W_full, cx, cy, target_h, target_w)
        firing_rates = get_activations(video_frames, retina, args.device)
        
        trials.append({
            "firing_rates": [rate.cpu() for rate in firing_rates],
            "target_video": torch.from_numpy(video_frames).float(),
            "crop": {"x": cx, "y": cy, "target_h": target_h, "target_w": target_w},
            "augmentation": {"angle": 0.0, "spatial_scale": 1.0, "temporal_scale": 1.0}
        })
        
        # Augmented Trials
        if args.augment:
            for _ in range(args.n_augs):
                angle = random.uniform(0, 360)
                s_scale = random.uniform(0.5, 2.0)
                t_scale = random.uniform(0.5, 2.0)
                
                aug_frames = apply_augmentation(full_video, cx, cy, target_h, target_w, angle, s_scale, t_scale)
                aug_rates = get_activations(aug_frames, retina, args.device)
                
                trials.append({
                    "firing_rates": [rate.cpu() for rate in aug_rates],
                    "target_video": torch.from_numpy(aug_frames).float(),
                    "crop": {"x": cx, "y": cy, "target_h": target_h, "target_w": target_w},
                    "augmentation": {"angle": angle, "spatial_scale": s_scale, "temporal_scale": t_scale}
                })

    # 6. Save output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSaving generated trials to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Save video_info
        v_info = f.create_group("video_info")
        v_info.attrs["path"] = video_path
        v_info.attrs["fps"] = fps
        v_info.attrs["shape"] = (T, H_full, W_full)
        
        # Save mosaic names
        f.attrs["mosaic_names"] = json.dumps(mosaic_names)
        
        # Save trials
        trials_grp = f.create_group("trials")
        for i, trial in enumerate(trials):
            trial_grp = trials_grp.create_group(str(i))
            
            # firing_rates is now a list of tensors
            fr_grp = trial_grp.create_group("firing_rates")
            for m_idx, rate in enumerate(trial["firing_rates"]):
                fr_grp.create_dataset(str(m_idx), data=rate.numpy(), compression="lzf")
                
            trial_grp.create_dataset("target_video", data=trial["target_video"].numpy(), compression="lzf")
            
            crop_grp = trial_grp.create_group("crop")
            for k, v in trial["crop"].items():
                crop_grp.attrs[k] = v
                
            aug_grp = trial_grp.create_group("augmentation")
            for k, v in trial["augmentation"].items():
                aug_grp.attrs[k] = v
                
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Generate RGC activations from a video using the Retina model.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file or directory of videos.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output activations (e.g., activations.h5) or directory if --video is a directory.")
    parser.add_argument("--x", type=int, default=None, help="Top-left X coordinate (column) for the spatial crop.")
    parser.add_argument("--y", type=int, default=None, help="Top-left Y coordinate (row) for the spatial crop.")
    parser.add_argument("--random-crops", type=int, default=0, help="If >0, ignores x/y and generates N random crops.")
    parser.add_argument("--augment", action="store_true", help="Enable randomized augmentations (rotation, spatial scale, temporal scale).")
    parser.add_argument("--n-augs", type=int, default=3, help="Number of augmented variations to generate per crop.")
    parser.add_argument("--cell-minibatch", type=int, default=500, help="Number of cells to process in a single vectorized batch.")
    parser.add_argument("--temp-batch", type=int, default=50, help="Number of temporal windows to process in a single vectorized batch.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")

    args = parser.parse_args()

    # 1. Load parameters
    params = u.read_params(args.params)
    video_params = params.get("video_parameters", {})
    target_h, target_w = video_params.get("frame_shape", [128, 128])
    
    if os.path.isdir(args.video):
        valid_exts = [".mp4", ".avi", ".mkv"]
        video_files = [f for f in os.listdir(args.video) if any(f.endswith(ext) for ext in valid_exts)]
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        print(f"Found {len(video_files)} videos in {args.video}.")
        for v in video_files:
            v_path = os.path.join(args.video, v)
            out_path = os.path.join(args.output, f"{os.path.splitext(v)[0]}_acts.h5")
            process_video(v_path, out_path, args, params, target_h, target_w)
    else:
        process_video(args.video, args.output, args, params, target_h, target_w)

if __name__ == "__main__":
    main()
