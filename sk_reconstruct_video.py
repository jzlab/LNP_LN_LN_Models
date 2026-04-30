import os
import argparse
import torch
import numpy as np
import cv2
import subprocess
import h5py
from tqdm import tqdm

from sk_utils import Utils as u
from sk_decoder import RetinaDecoder

def setup_decoder(params, training_params, n_cells_per_mosaic, device="cpu", weights_path=None):
    print("Initializing Decoder model...")
    input_window = training_params.get("input_window_size", 1)
    
    decoder_params = {
        "n_cells_per_mosaic": n_cells_per_mosaic,
        "frame_shape": params.get("video_parameters", {}).get("frame_shape", [128, 128]),
        "num_blocks": training_params.get("num_blocks", 4),
        "num_kernels": training_params.get("num_kernels", 64),
        "bias": training_params.get("bias", False)
    }
    
    model = RetinaDecoder(decoder_params).to(device)
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model, input_window

def decode_activations(firing_rates_list, decoder, input_window, original_T, device="cpu"):
    """
    firing_rates_list: list of tensors, each (max_cells, n_windows)
    """
    n_windows = firing_rates_list[0].shape[1]
    
    # Valid indices where we have sufficient context:
    valid_indices = np.arange(input_window - 1, n_windows)
    retina_win_size = original_T - n_windows + 1
    
    decoded_frames = []
    with torch.no_grad():
        batch_size = 16
        for start_idx in range(0, len(valid_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(valid_indices))
            
            x_batch_list = [[] for _ in range(len(firing_rates_list))]
            for i in range(start_idx, end_idx):
                t = valid_indices[i]
                for m_idx in range(len(firing_rates_list)):
                    x = firing_rates_list[m_idx][:, t - (input_window - 1) : t + 1]
                    x_batch_list[m_idx].append(x)
            
            x_batch_list = [torch.stack(b).to(device) for b in x_batch_list]
            out = decoder(x_batch_list) # (B, 1, H, W)
            out = out.squeeze(1).cpu().numpy() # (B, H, W)
            decoded_frames.extend(out)
            
    start_frame_idx = input_window - 1 + retina_win_size - 1
    return np.array(decoded_frames), start_frame_idx

def pad_dimensions(H, W, patch_h, patch_w):
    pad_h = (patch_h - (H % patch_h)) % patch_h if H % patch_h != 0 else 0
    pad_w = (patch_w - (W % patch_w)) % patch_w if W % patch_w != 0 else 0
    return H + pad_h, W + pad_w

def main():
    parser = argparse.ArgumentParser(description="Reconstruct full video by chunking")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--training-params", type=str, default="params_training.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to decoder weights (.pt)")
    parser.add_argument("--activations", type=str, default=None, help="Path to pre-generated activations (.pt) to skip Phase 1")
    parser.add_argument("--output-activations", type=str, default="full_video_activations.pt")
    parser.add_argument("--output-video", type=str, default="reconstructed.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--zero-celltype", type=int, nargs="+", default=None, help="Indices of celltypes to zero out before decoding")
    
    args = parser.parse_args()
    
    params = u.read_params(args.params)
    train_params = u.read_params(args.training_params)
    video_params = params.get("video_parameters", {})
    patch_h, patch_w = video_params.get("frame_shape", [128, 128])
    
    print(f"Loading baseline video info from {args.video}...")
    # Just need dimensions, so we fetch it quickly
    _, props = u.read_video(args.video)
    fps, H_full, W_full, T = props["fps"], int(props["height"]), int(props["width"]), int(props["total_frames"])
    
    print(f"Original video: {T} frames, {H_full}x{W_full} resolution at {fps} FPS.")
    
    H_pad, W_pad = pad_dimensions(H_full, W_full, patch_h, patch_w)
    print(f"Padded canvas target: {T} frames, {H_pad}x{W_pad}.")
    
    rows = H_pad // patch_h
    cols = W_pad // patch_w
    print(f"Tiling grid: {rows} rows x {cols} cols (Total {rows*cols} patches).")
    
    # 1. Generate & Load Activations using os.system via subprocess
    activations_dict = {}
    
    if args.activations and os.path.exists(args.activations):
        print(f"\n[Phase 1] Skipping generation. Loading provided activations from {args.activations}...")
        data = torch.load(args.activations, map_location="cpu")
        activations_dict = data["activations"]
    else:
        print("\n[Phase 1] Generating Retina activations for all spatial patches using sk_generate_activations.py...")
        
        # Optional flags for batching. The user requested we use their inputs from before.
        cell_minibatch = 500
        temp_batch = 64
        
        # Prepare python execution (using current sys.executable or plain 'python')
        import sys
        python_exec = sys.executable
        
        for r in tqdm(range(rows)):
            for c in range(cols):
                y_start = r * patch_h
                x_start = c * patch_w
                
                tmp_act_path = f"tmp_act_r{r}_c{c}.h5"
                
                # Execute the script
                cmd = [
                    python_exec, "sk_generate_activations.py",
                    "--video", args.video,
                    "--params", args.params,
                    "--output", tmp_act_path,
                    "--x", str(x_start),
                    "--y", str(y_start),
                    "--cell-minibatch", str(cell_minibatch),
                    "--temp-batch", str(temp_batch),
                    "--device", args.device
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error running sk_generate_activations for patch ({r}, {c})!")
                    print(result.stderr)
                    return
                
                # Load the generated activations from HDF5
                with h5py.File(tmp_act_path, 'r') as f:
                    trials_grp = f["trials"]
                    first_k = list(trials_grp.keys())[0]
                    fr_grp = trials_grp[first_k]["firing_rates"]
                    fr_list = []
                    for m_idx in range(len(fr_grp.keys())):
                        fr_list.append(torch.from_numpy(fr_grp[str(m_idx)][:]))
                    activations_dict[(r, c)] = fr_list
                
                # Remove tmp file to save disk space
                os.remove(tmp_act_path)
                
        # Save the consolidated activations
        print(f"\nSaving consolidated activations to {args.output_activations}...")
        torch.save({
            "activations": activations_dict,
            "video_info": {"path": args.video, "fps": fps, "original_shape": (T, H_full, W_full), "padded_shape": (T, H_pad, W_pad)},
        }, args.output_activations)
    
    # 2. Setup Decoder & Decode
    sample_act = activations_dict[(0,0)]
    input_window = train_params.get("input_window_size", 1)
    
    n_cells_per_mosaic = [act.shape[0] * input_window for act in sample_act]
    decoder, _ = setup_decoder(params, train_params, n_cells_per_mosaic=n_cells_per_mosaic, device=args.device, weights_path=args.weights)
    
    n_windows = sample_act[0].shape[1]
    decoded_T = n_windows - input_window + 1
    output_buffer = np.zeros((decoded_T, H_pad, W_pad), dtype=np.float32)
    
    print("\n[Phase 2] Decoding activations to reconstruct video patches...")
    for r in tqdm(range(rows)):
        for c in range(cols):
            y_start = r * patch_h
            y_end = y_start + patch_h
            x_start = c * patch_w
            x_end = x_start + patch_w
            
            firing_rates_list = activations_dict[(r, c)]
            if args.zero_celltype is not None:
                for celltype in args.zero_celltype:
                    firing_rates_list[celltype][:, :] = 0.0
                
            decoded_patch, start_frame_idx = decode_activations(firing_rates_list, decoder, input_window, T, device=args.device)
            output_buffer[:, y_start:y_end, x_start:x_end] = decoded_patch

    # 3. Stitching & Export
    print(f"\nCropping reconstructed video back to {H_full}x{W_full}...")
    reconstructed = output_buffer[:, :H_full, :W_full]
    
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    
    out_path = args.output_video
    print(f"Saving output video to {out_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Alternative: 'mp4v' or 'avc1'
    out_video = cv2.VideoWriter(out_path, fourcc, int(fps), (W_full, H_full))
    
    for i in range(len(reconstructed_uint8)):
        out_video.write(cv2.cvtColor(reconstructed_uint8[i], cv2.COLOR_GRAY2BGR))
    
    out_video.release()
    print("Finished! Reconstructed video generated successfully.")

if __name__ == "__main__":
    main()
