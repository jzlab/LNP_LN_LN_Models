import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sk_utils import Utils as u
from sk_decoder import Decoder
from sk_generate_activations import load_and_crop_video

class RGCDataset(Dataset):
    """
    Dataset that pairs RGC activations with their corresponding video frames.
    """
    def __init__(self, activations_path, training_params_path):
        # 1. Load activations and metadata
        data = torch.load(activations_path)
        self.firing_rates = data["firing_rates"] # (n_mosaics, n_cells, n_windows)
        self.video_info = data["video_info"]
        
        # 2. Load and crop original video
        # Re-use the cropping logic to ensure exact parity
        x_start = self.video_info.get("x", 0)
        y_start = self.video_info.get("y", 0)
        target_h, target_w = self.video_info["shape"][1], self.video_info["shape"][2]
        
        self.video_frames, _, _ = load_and_crop_video(
            self.video_info["path"], 
            x_start, y_start, 
            target_h, target_w
        )
        
        # Load training params for alignment config
        train_cfg = u.read_params(training_params_path)
        self.input_window = train_cfg.get("input_window_size", 1)
        
        # Determine temporal alignment
        n_windows = self.firing_rates.shape[2]
        n_frames = len(self.video_frames)
        self.retina_win_size = n_frames - n_windows + 1
        
        # We can only train on windows where we have at least input_window of context
        self.valid_indices = np.arange(self.input_window - 1, n_windows)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual activation index
        t = self.valid_indices[idx]
        
        # Input: sliding window of activations (K, n_mosaics, n_cells)
        x = self.firing_rates[:, :, t - (self.input_window - 1) : t + 1]
        
        # Target: The 'last' frame of the L-frame retina window at time t
        target_idx = t + self.retina_win_size - 1
        y = torch.from_numpy(self.video_frames[target_idx]).unsqueeze(0).float()
        
        return x, y

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
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model Initialization
    n_mosaics, max_n_cells, _ = dataset.firing_rates.shape
    decoder_params = {
        "n_cells": n_mosaics * max_n_cells * dataset.input_window,
        "frame_shape": dataset.video_info["shape"][1:],
        "num_blocks": train_cfg.get("num_blocks", 4),
        "num_kernels": train_cfg.get("num_kernels", 64),
        "bias": train_cfg.get("bias", False)
    }
    
    model = Decoder(decoder_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {num_params:,}")

    # 4. Training Loop
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
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
        
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        epoch_pbar.set_postfix({"Train MSE": f"{avg_train:.6f}", "Val MSE": f"{avg_val:.6f}"})
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output)
            print(f"\nNew best model saved to {args.output} (Val MSE: {avg_val:.6f})")

    print("\nTraining Complete.")

def main():
    parser = argparse.ArgumentParser(description="Train the Decoder model on RGC activations.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the .pt file containing RGC activations.")
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
