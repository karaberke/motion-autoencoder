import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import CMUMotionDataset
from visualization import *

class MotionAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Motion Data as described in the paper
    "Learning Motion Manifolds with Convolutional Autoencoders"
    """
    def __init__(self, input_dim=63):
        super(MotionAutoencoder, self).__init__()
        
        # Encoder network with 3 convolutional layers
        # TODO: Complete the encoder architecture.
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 192, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(192, 256, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # The decoder needs to upsample to restore the original dimensions
        # TODO: Complete the decoder architecture.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(256, 192, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(192, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(128, input_dim, kernel_size=25, padding=12),
        )
    
    def encode(self, x):
        """Project onto the manifold (Φ operation)"""
        return self.encoder(x)
    
    def decode(self, z):
        """Inverse projection from the manifold (Φ† operation)"""
        return self.decoder(z)
    
    def forward(self, x, corrupt_input=False, corruption_prob=0.1):
        """Forward pass with optional denoising"""
        if corrupt_input and self.training:
            # Create corruption mask (randomly set values to zero with probability corruption_prob)
            mask = torch.bernoulli(torch.ones_like(x) * (1 - corruption_prob))
            x_corrupted = x * mask
        else:
            x_corrupted = x
            
        # TODO: Implement the forward pass for the autoencoder.
        z = self.encode(x_corrupted)
        return self.decode(z), z


class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Convolutional Autoencoder"""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 0.5,
        fine_tune_lr: float = 0.01,
        sparsity_weight: float = 0.01,
        window_size: int = 160,
        val_split: float = 0.1,
        device: str = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.sparsity_weight = sparsity_weight
        self.window_size = window_size
        self.val_split = val_split
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load dataset
        self._load_dataset()
        
        # Initialize model
        self._init_model()
        
    def _load_dataset(self):
        """Load the CMU Motion dataset and create training/validation splits"""
        # Create dataset
        from dataloader import CMUMotionDataset
        
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )
        
        # Split into training and validation sets
        val_size = int(self.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        # Get mean and std for normalization
        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()
        
    def _init_model(self):
        """Initialize the motion autoencoder model"""
        # Get a sample to determine dimensions
        sample = self.dataset[0]
        
        # Get the flattened positions with velocities for the paper's approach
        # We'll use positions_flat which has global transforms removed
        if "positions_flat" in sample:
            positions_flat = sample["positions_flat"]
            
            # Check if we need to add velocities to the input 
            # The paper mentions including rotational velocity around Y and translational velocity in XZ
            if "trans_vel_xz" in sample and "rot_vel_y" in sample:
                # Get velocity data
                trans_vel_xz = sample["trans_vel_xz"]
                rot_vel_y = sample["rot_vel_y"]
                
                # Create input with features as separate channels (matches paper description)
                input_dim = positions_flat.shape[1] + trans_vel_xz.shape[1] + 1  # positions + trans_vel_xz + rot_vel_y
                print(f"Input includes positions ({positions_flat.shape[1]} dims) and velocities ({trans_vel_xz.shape[1] + 1} dims)")
            else:
                # Just use positions if velocities aren't available
                input_dim = positions_flat.shape[1]
                print(f"Input only includes positions ({input_dim} dims)")
        else:
            # Fallback to original positions if flattened positions aren't available
            positions = sample["positions"]
            # Calculate input dimension from sample
            # For the paper's approach, we need to flatten joints and dimensions
            # positions is [time, joints, 3], we need to get joints*3
            input_dim = positions.shape[1] * positions.shape[2]
            print(f"Using fallback input dimension: {input_dim}")
        
        # Create model
        self.input_dim = input_dim
        self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
        print(f"Created model with input dimension: {input_dim}")
        
    def train(self):
        """Train the motion autoencoder in two phases: initial training and fine-tuning"""
        # TODO: Implement the training phases for the motion autoencoder.
        # There are multiple training phases described in the paper, you can implement them in this function. You can implement the training phases as separate functions if you prefer or you can use several parameters to combine them to one function.
        initial_stats = self._train_phase(self.epochs, self.learning_rate, 0.1, self.sparsity_weight, "initial")
        finetune_stats = self._train_phase(self.fine_tune_epochs, self.fine_tune_lr, 0.0, self.sparsity_weight, "finetune")
        
        # Combine statistics: you can return stats here as a dictionary and use our plotting function to plot the training curves.
        all_stats = {"initial": initial_stats, "finetune": finetune_stats}
        
        # Save training statistics
        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)
            
        # Save final model
        self._save_model()
            
        # Save normalization parameters
        self._save_normalization_params()
        
        # Plot training curves
        self._plot_training_curves(all_stats)
        
        return all_stats

    # This is a sample of what you can use in the training phase. You are not required to follow it as long as you can provide the training statistics we required.
    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        """Train the model for a specific phase (initial training or fine-tuning)"""
        print(f"\n===== {phase_name.capitalize()} Training Phase =====")
        
        # TODO: Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        vel_mean = torch.tensor(self.dataset.vel_mean, dtype=torch.float32)
        vel_std = torch.tensor(self.dataset.vel_std, dtype=torch.float32)
        
        
        # Training stats
        stats = {
            # TODO: Define and accumulate the training/validation statistics to track.
            "train_loss": [],
            "val_loss": [],
        }
        
        # Track training checkpoints
        best_val_loss = float("inf")
        
        # Train for specified epochs
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                # TODO: Implement the training loop for the motion autoencoder.
                vel = (torch.cat([batch["trans_vel_xz"], batch["rot_vel_y"].unsqueeze(-1)], dim=-1) - vel_mean) / vel_std
                x = torch.cat([batch["positions_normalized_flat"], vel], dim=-1).permute(0, 2, 1).to(self.device)
                recon, z = self.model(x, corrupt_input=corruption_prob > 0, corruption_prob=corruption_prob)
                recon_loss = criterion(recon, x)
                loss = recon_loss + sparsity_weight * z.abs().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.shape[0] / len(self.train_dataset)
                
                # Update progress bar
                progress_bar.set_postfix({
                    # You can set what you want to track with the progress bar.
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                })
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in progress_bar:
                    # TODO: Implement the validation loop for the motion autoencoder.
                    vel = (torch.cat([batch["trans_vel_xz"], batch["rot_vel_y"].unsqueeze(-1)], dim=-1) - vel_mean) / vel_std
                    x = torch.cat([batch["positions_normalized_flat"], vel], dim=-1).permute(0, 2, 1).to(self.device)
                    recon, _ = self.model(x)
                    val_loss += criterion(recon, x).item() * x.shape[0] / len(self.val_dataset)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        # You can set what you want to track with the progress bar.
                        "val_loss": f"{val_loss:.4f}",
                    })
            
            stats["train_loss"].append(train_loss)
            stats["val_loss"].append(val_loss)

            # Save if best model. You can also save checkpoints each epoch if needed.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, end=f'_valloss_{val_loss:.6f}', phase_name=phase_name)
                print(f"  Saved checkpoint with val_loss: {val_loss:.6f}")
        
        return stats
    
    def _save_checkpoint(self, epoch, end, phase_name):
        """Save a model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt"
        )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }, checkpoint_path)
    
    def _save_model(self):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), model_path)
        with open(os.path.join(self.output_dir, "models", "model_config.json"), "w") as f:
            json.dump({"input_dim": self.input_dim}, f)
        print(f"Model saved to {model_path}")
    
    def _save_normalization_params(self):
        """Save normalization parameters for inference if needed"""
        norm_data = {
            "mean_pose": self.mean_pose.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "joint_names": self.joint_names,
            "joint_parents": self.joint_parents
        }
        
        np.save(os.path.join(self.output_dir, "normalization.npy"), norm_data)
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npy")
    
    def _plot_training_curves(self, stats):
        """Plot training curves for one or more training phases"""
        if not isinstance(stats, dict) or not stats:
            raise ValueError("Return a non-empty statistics dictionary from your training implementation.")
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
            
        n_p = len(list(stats.keys()))
        plt.figure(figsize=(12, 4 * n_p))
        # Multiple training phases
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch")
            plt.ylabel("Statistics")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


class MotionManifoldSynthesizer:
    """Synthesizer for generating, fixing, and analyzing motion using the learned manifold"""
    def __init__(
        self,
        model_path: str,
        dataset: CMUMotionDataset,
        device: str = None
    ):
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load normalization parameters
        self._load_normalization(dataset)
        
        # Load model
        self._load_model(model_path)
    
    def _load_normalization(self, dataset: CMUMotionDataset):
        """Load normalization parameters from dataset"""
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)
        self.std = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)
        self.joint_names = dataset.joint_names
        self.joint_parents = dataset.joint_parents
    
    def _load_model(self, model_path):
        """Load trained model"""
        if os.path.exists(model_path):
            # Determine input dimension from the model's saved state
            model_state = torch.load(model_path, map_location=self.device)
            
            # Try to infer input dimension from the first layer weights
            first_layer_weight = None
            for key in model_state.keys():
                if 'encoder.0.weight' in key:
                    first_layer_weight = model_state[key]
                    break
            
            config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    input_dim = int(json.load(f)["input_dim"])
            elif first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
                print(f"Inferred input dimension {input_dim} from model weights")
            else:
                # Fallback to calculating from mean_pose if we can't find the weights
                input_dim = self.mean_pose.shape[0] * self.mean_pose.shape[1]
                print(f"Using fallback input dimension: {input_dim}")
                
            # Create model
            self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
            
            # Load weights
            self.model.load_state_dict(model_state)
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        """
        Fix corrupted motion by projecting onto the manifold and recovering global motion
        
        Args:
            motion: batched dataset dictionary; positions has shape [batch_size, time_steps, joints, dims]
            corruption_type: Type of corruption to apply ('zero', 'noise', or 'missing')
            corruption_params: Parameters for corruption
                    
        Returns:
            Tuple of (corrupted_motion, fixed_motion)
        """
        positions = motion['positions'].to(self.device)
        # Store original shape
        original_shape = positions.shape
        batch_size, time_steps, joints, dims = original_shape
        
        # Apply corruption if not already corrupted
        if corruption_params is not None:
            corrupted_motion = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted_motion = positions.clone()
            
        # TODO: Fix the corrupted motion by your model.
        # HINT: You need to normalize the corrupted motion and then project it onto the manifold using the model. Then unnormalize the fixed motion.
        # HINT: If you like, you can recover global motion by calling recover_global_motion in dataloader.py
        x = ((corrupted_motion - self.mean_pose) / self.std).reshape(batch_size, time_steps, joints * dims)
        x = torch.cat([x, torch.zeros(batch_size, time_steps, 3, device=self.device)], dim=-1).permute(0, 2, 1)
        with torch.no_grad():
            y = self.model.decode(self.model.encode(x))
        fixed_motion = y.permute(0, 2, 1)[..., :joints * dims].reshape(original_shape) * self.std + self.mean_pose
        
        # Return corrupted motion and fixed motion with global transform applied
        return corrupted_motion, fixed_motion
    
    def _apply_corruption(self, motion, corruption_type, params):
        """Apply corruption to motion data"""
        corrupted = motion.clone()
        
        if corruption_type == 'zero':
            # Randomly set values to zero
            prob = params.get('prob', 0.5)
            mask = torch.bernoulli(torch.ones_like(corrupted) * (1 - prob))
            corrupted = corrupted * mask
            
        elif corruption_type == 'noise':
            # Add Gaussian noise
            noise_scale = params.get('scale', 0.1)
            noise = torch.randn_like(corrupted) * noise_scale
            corrupted = corrupted + noise
            
        elif corruption_type == 'missing':
            # Set specific joint to zero
            joint_idx = params.get('joint_idx', 0)
            corrupted[:, :, joint_idx, :] = 0.0
            
        return corrupted
    
    def interpolate_motions(self, motion1, motion2, t):
        """
        Interpolate between two motions on the manifold, handling global transforms
        
        Args:
            motion1: tensor of shape [batch_size, time_steps, joints, dims]
            motion2: tensor of shape [batch_size, time_steps, joints, dims]
            t: Interpolation parameter (0 to 1)
                    
        Returns:
            Interpolated motion as tensor of shape [batch_size, time_steps, joints, dims]
        """
        # TODO: Implement motion interpolation on the manifold.
        # HINT: You can use the model to project the motions onto the manifold and then interpolate in the latent space.
        # HINT: To simplify implementation, you could only implement the version where both motions are local motions (without global transforms).
        batch_size, time_steps, joints, dims = motion1.shape
        x1 = ((motion1.to(self.device) - self.mean_pose) / self.std).reshape(batch_size, time_steps, joints * dims)
        x2 = ((motion2.to(self.device) - self.mean_pose) / self.std).reshape(batch_size, time_steps, joints * dims)
        zeros = torch.zeros(batch_size, time_steps, 3, device=self.device)
        x1 = torch.cat([x1, zeros], dim=-1).permute(0, 2, 1)
        x2 = torch.cat([x2, zeros], dim=-1).permute(0, 2, 1)
        with torch.no_grad():
            z = (1 - t) * self.model.encode(x1) + t * self.model.encode(x2)
            y = self.model.decode(z)
        return y.permute(0, 2, 1)[..., :joints * dims].reshape(motion1.shape) * self.std + self.mean_pose
    
    # You can add more functions for Extra Credit.

    def complete_motion(self, motion, known, iterations=30, lr=0.01):
        positions = motion.to(self.device)
        batch_size, time_steps, joints, dims = positions.shape
        known_idx = np.flatnonzero(known.cpu().numpy())
        flat = positions.reshape(batch_size, time_steps, -1).cpu().numpy()
        init = np.stack([np.stack([np.interp(np.arange(time_steps), known_idx, f[known_idx, c]) for c in range(f.shape[1])], axis=1) for f in flat])
        init = torch.tensor(init, dtype=torch.float32, device=self.device).reshape(positions.shape)
        mask = known.to(self.device).view(1, time_steps, 1, 1)
        target = (positions - self.mean_pose) / self.std
        x = (init - self.mean_pose) / self.std
        x = torch.cat([x.reshape(batch_size, time_steps, -1), torch.zeros(batch_size, time_steps, 3, device=self.device)], dim=-1).permute(0, 2, 1)
        with torch.no_grad():
            z = self.model.encode(x)
        z.requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        for _ in range(iterations):
            y = self.model.decode(z).permute(0, 2, 1)[..., :joints * dims].reshape(positions.shape)
            loss = ((y - target)[:, known.to(self.device)] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z.clamp_(min=0)
        with torch.no_grad():
            y = self.model.decode(z).permute(0, 2, 1)[..., :joints * dims].reshape(positions.shape)
            recon = y * self.std + self.mean_pose
        return torch.where(mask, positions, recon)

    def transfer_style(self, content, style, style_weight=10.0, iterations=500, lr=0.01):
        def encode(positions):
            positions = positions.to(self.device)
            batch_size, time_steps, joints, dims = positions.shape
            x = ((positions - self.mean_pose) / self.std).reshape(batch_size, time_steps, joints * dims)
            x = torch.cat([x, torch.zeros(batch_size, time_steps, 3, device=self.device)], dim=-1).permute(0, 2, 1)
            with torch.no_grad():
                return self.model.encode(x)

        def gram(z):
            return z @ z.transpose(1, 2) / z.shape[-1]

        z_content = encode(content)
        style_gram = gram(encode(style))
        z = z_content.clone().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        for _ in range(iterations):
            loss = ((z - z_content) ** 2).mean() + style_weight * ((gram(z) - style_gram) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z.clamp_(min=0)
        with torch.no_grad():
            y = self.model.decode(z)
        batch_size, time_steps, joints, dims = content.shape
        return y.permute(0, 2, 1)[..., :joints * dims].reshape(content.shape).cpu() * self.std.cpu() + self.mean_pose.cpu()


def main():
    """Example usage of the motion manifold training"""
    
    # Training parameters
    data_dir = "./cmu-mocap"
    output_dir = "./output/ae"
    
    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        epochs=25,              # Initial training epochs
        fine_tune_epochs=25,    # Fine-tuning epochs
        learning_rate=0.001,    # Initial learning rate
        fine_tune_lr=0.001,     # Fine-tuning learning rate
        sparsity_weight=0.01,   # Sparsity constraint weight
        window_size=160,        # Window size (as in paper)
        val_split=0.1,          # Validation split
        device="mps"
    )
    
    # Train the model
    trainer.train()
    
    # For inference, you can load the dataset and model and use the synthesizer for different tasks. 
    # You can also use the visualization functions to visualize the results following examples in dataloader.py.


if __name__ == "__main__":
    main()
    
    