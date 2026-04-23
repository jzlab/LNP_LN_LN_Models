import os
import glob
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sk_utils import Utils as u
from sk_decoder import RetinaDecoder

class RGCDataset(Dataset):
    """
    Dataset that pairs RGC activations with their corresponding video frames.
    """
    def __init__(self, activations_path, training_params_path):
        import h5py
        
        train_cfg = u.read_params(training_params_path)
        self.input_window = train_cfg.get("input_window_size", 1)
        
        if os.path.isdir(activations_path):
            file_paths = glob.glob(os.path.join(activations_path, "*.h5"))
        else:
            file_paths = [activations_path]
            
        if not file_paths:
            raise ValueError(f"No .h5 files found for path: {activations_path}")
            
        self.open_files = []
        self.index_map = []
        
        for fp in file_paths:
            f = h5py.File(fp, 'r')
            file_idx = len(self.open_files)
            self.open_files.append(f)
            
            if "trials" not in f:
                print(f"Skipping {fp}: not in trials format.")
                continue
                
            trials_grp = f["trials"]
            
            for local_idx in trials_grp.keys():
                trial_grp = trials_grp[local_idx]
                
                fr_shape = trial_grp["firing_rates"].shape
                vf_shape = trial_grp["target_video"].shape
                
                n_windows = fr_shape[2]
                T = vf_shape[0]
                retina_win_size = T - n_windows + 1
                
                for t in range(self.input_window - 1, n_windows):
                    self.index_map.append((file_idx, local_idx, t, retina_win_size))
                    
        print(f"Dataset compiled: {len(self.index_map)} valid spatiotemporal samples from {len(file_paths)} files.")
        
        if len(self.index_map) == 0:
             raise ValueError("No valid trials found in the provided activations.")
        
        # Save dimensions for model initialization
        f0 = self.open_files[0]
        first_trial = list(f0["trials"].keys())[0]
        self.n_mosaics, self.max_n_cells, _ = f0["trials"][first_trial]["firing_rates"].shape
        self.frame_shape = f0["trials"][first_trial]["target_video"].shape[1:]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx, t, retina_win_size = self.index_map[idx]
        h5_file = self.open_files[file_idx]
        
        fr_ds = h5_file[f"trials/{local_idx}/firing_rates"]
        vf_ds = h5_file[f"trials/{local_idx}/target_video"]
        
        x_np = fr_ds[:, :, t - (self.input_window - 1) : t + 1]
        
        target_idx = t + retina_win_size - 1
        y_np = vf_ds[target_idx]
        
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).unsqueeze(0).float()
        
        return x, y
        
    def __del__(self):
        for f in getattr(self, 'open_files', []):
            try:
                f.close()
            except:
                pass

def train(args):
    # 1. Load Configurations
    model_params = u.read_params(args.params)
    train_cfg = u.read_params(args.training_params)

    # CLI overrides
    device = args.device if args.device else train_cfg.get("device", "cpu")
    epochs = args.epochs if args.epochs else train_cfg.get("epochs", 50)
    batch_size = args.batch_size if args.batch_size else train_cfg.get("batch_size", 16)
    lr = args.lr if args.lr else train_cfg.get("learning_rate", 1e-4)

    print(f"Training on device: {device}")

    # 2. Data Preparation
    print(f"Loading data from {args.activations}...")
    dataset = RGCDataset(args.activations, args.training_params)
    
    # Validation split
    val_split = train_cfg.get("validation_split", 0.1)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model Initialization
    n_mosaics, max_n_cells = dataset.n_mosaics, dataset.max_n_cells
    decoder_params = {
        "n_cells": n_mosaics * max_n_cells * dataset.input_window,
        "frame_shape": dataset.frame_shape,
        "num_blocks": train_cfg.get("num_blocks", 4),
        "num_kernels": train_cfg.get("num_kernels", 64),
        "bias": train_cfg.get("bias", False)
    }
    
    model = RetinaDecoder(decoder_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {num_params:,}\n")

    # 4. Training Loop
    best_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", dynamic_ncols=False, leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        # iterate through the batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)
        
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        epoch_pbar.set_postfix({"Train MSE": f"{avg_train:.6f}", "Val MSE": f"{avg_val:.6f}", "Best Val MSE": f"{best_val_loss:.6f} @ Epoch {best_epoch}"})
        
        if avg_val < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = avg_val
            save_dict = {
                'model_state_dict': model.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }
            torch.save(save_dict, args.output)

    print("\nTraining Complete.")
    
    # Save the final model and full history
    last_output = args.output.replace(".pt", "_final.pt") if ".pt" in args.output else args.output + "_final.pt"
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }
    torch.save(save_dict, last_output)
    print(f"Final model and training history saved to {last_output}\n")

def main():
    parser = argparse.ArgumentParser(description="Train the Decoder model on RGC activations.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the .h5 file or directory containing RGC activations.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to the retina model params.yaml.")
    parser.add_argument("--training-params", type=str, default="training_params.yaml", help="Path to training_params.yaml.")
    parser.add_argument("--output", type=str, default="best_decoder.pt", help="Path to save the best model weights.")
    parser.add_argument("--device", type=str, help="Device to use (cpu, cuda). Overrides training_params.yaml.")
    parser.add_argument("--epochs", type=int, help="Number of epochs. Overrides training_params.yaml.")
    parser.add_argument("--batch-size", type=int, help="Batch size. Overrides training_params.yaml.")
    parser.add_argument("--lr", type=float, help="Learning rate. Overrides training_params.yaml.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
