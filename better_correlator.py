#!/usr/bin/env python3
# src/models/simple_e3nn_correlator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import e3nn
from e3nn import o3

# Import your data structures
from fung_data_parser import ThreeDCTDataParser, CorrelationSession, TransformationParams

class SimpleE3NNCorrelator(nn.Module):
    """Simple E3NN-based correlator without graphs - just using tensor products and linear layers"""
    
    def __init__(self, max_fiducials: int = 25):
        super().__init__()
        
        self.max_fiducials = max_fiducials
        
        # Define irreps for our data
        # Each fiducial is a 3D position (1 vector)
        self.irreps_input = o3.Irreps("1x1e")  # 1 vector (3D position)
        self.irreps_hidden = o3.Irreps("16x0e + 8x1e + 4x2e")  # mixed irreps for expressivity
        self.irreps_output = o3.Irreps("1x0e")  # scalar per coordinate
        
        # E3NN layers
        # First layer: process individual 3D positions
        self.linear1 = o3.Linear(
            irreps_in=self.irreps_input,
            irreps_out=self.irreps_hidden
        )
        
        # Second layer: mix the features
        self.linear2 = o3.Linear(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_hidden
        )
        
        # Third layer: extract scalars for final processing
        self.linear3 = o3.Linear(
            irreps_in=self.irreps_hidden,
            irreps_out="32x0e"  # 32 scalars
        )
        
        # Final MLP to map to 2D coordinates (breaks equivariance intentionally)
        self.final_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # Output 2D coordinates
        )
        
        # Global transformation prediction from all fiducials
        self.global_mlp = nn.Sequential(
            nn.Linear(32 * max_fiducials, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Transformation parameters
        )
        
    def forward(self, pos_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            pos_3d: [batch_size, num_fiducials, 3] - 3D positions
            
        Returns:
            pos_2d: [batch_size, num_fiducials, 2] - predicted 2D positions
            transform_params: [batch_size, 16] - transformation parameters
        """
        batch_size, num_fiducials, _ = pos_3d.shape
        
        # Process each fiducial independently with E3NN layers
        # Reshape to process all fiducials in all batches at once
        pos_flat = pos_3d.view(-1, 3)  # [batch_size * num_fiducials, 3]
        
        # E3NN processing (maintains equivariance)
        # Input is 3D positions which transform as vectors (1e irrep)
        features = self.linear1(pos_flat.unsqueeze(1))  # Add channel dim for vector
        features = self.linear2(features)
        features = self.linear3(features)
        
        # Extract scalar features (32 scalars per fiducial)
        scalar_features = features.squeeze(1)  # Remove channel dim
        
        # Reshape back to batch form
        scalar_features = scalar_features.view(batch_size, num_fiducials, 32)
        
        # Final projection to 2D (intentionally breaks equivariance)
        pos_2d = self.final_mlp(scalar_features)
        
        # Global transformation prediction
        # Flatten all features to predict global transformation
        global_features = scalar_features.view(batch_size, -1)
        
        # Pad or truncate to fixed size for global MLP
        if global_features.size(1) < 32 * self.max_fiducials:
            # Pad with zeros
            padding = torch.zeros(batch_size, 32 * self.max_fiducials - global_features.size(1), 
                                device=global_features.device)
            global_features = torch.cat([global_features, padding], dim=1)
        else:
            # Truncate
            global_features = global_features[:, :32 * self.max_fiducials]
        
        transform_params = self.global_mlp(global_features)
        
        return pos_2d, transform_params


class FixedFiducialDataset(torch.utils.data.Dataset):
    """Fixed dataset that handles individual fiducials properly"""
    
    def __init__(self, sessions: List[CorrelationSession], max_fiducials: int = 25):
        self.sessions = sessions
        self.max_fiducials = max_fiducials
        self.samples = []
        
        # Process each session, treating each fiducial as a separate sample
        for session in sessions:
            transform_params = self._convert_transform_to_vector(session.transformation)
            
            for fid in session.fiducial_pairs:
                self.samples.append({
                    'pos_3d': fid.initial_3d,
                    'pos_2d': fid.final_2d,
                    'transform_params': transform_params
                })
        
        # Calculate normalization statistics
        self._calculate_normalization()
    
    def _convert_transform_to_vector(self, transform: TransformationParams) -> np.ndarray:
        """Convert transformation to a simple vector representation"""
        # Just use the essential parameters
        params = np.concatenate([
            transform.rotation_euler / 180.0,  # Normalize angles
            [transform.scale - 1.0],  # Center scale around 0
            transform.translation_center / 1000.0,  # Normalize translation
            transform.center_point / 1000.0  # Normalize center
        ])
        return params
    
    def _calculate_normalization(self):
        """Calculate normalization statistics"""
        all_3d = np.array([item['pos_3d'] for item in self.samples])
        all_2d = np.array([item['pos_2d'] for item in self.samples])
        
        self.mean_3d = np.mean(all_3d, axis=0)
        self.std_3d = np.std(all_3d, axis=0) + 1e-8
        self.mean_2d = np.mean(all_2d, axis=0)
        self.std_2d = np.std(all_2d, axis=0) + 1e-8
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Normalize coordinates
        pos_3d = (sample['pos_3d'] - self.mean_3d) / self.std_3d
        pos_2d = (sample['pos_2d'] - self.mean_2d) / self.std_2d
        
        return {
            'pos_3d': torch.tensor(pos_3d, dtype=torch.float32),
            'pos_2d': torch.tensor(pos_2d, dtype=torch.float32),
            'transform_params': torch.tensor(sample['transform_params'], dtype=torch.float32)
        }


def train_simple_e3nn(
    data_dir: str,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu"
):
    """Train the simple E3NN model"""
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load data
    parser = ThreeDCTDataParser(data_dir)
    sessions = parser.load_multiple_sessions()
    
    if not sessions:
        raise ValueError(f"No sessions found in {data_dir}")
    
    print(f"Loaded {len(sessions)} sessions")
    
    # Create dataset with individual fiducials
    dataset = FixedFiducialDataset(sessions)
    print(f"Total fiducials: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = SimpleE3NNCorrelator().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Training loop
    model.train()
    loss_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        train_losses = []
        for batch in train_loader:
            pos_3d = batch['pos_3d'].to(device).unsqueeze(1)  # Add fiducial dimension
            pos_2d = batch['pos_2d'].to(device).unsqueeze(1)  # Add fiducial dimension
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_2d, _ = model(pos_3d)
            
            # Calculate loss
            loss = F.mse_loss(pred_2d, pos_2d)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pos_3d = batch['pos_3d'].to(device).unsqueeze(1)  # Add fiducial dimension
                pos_2d = batch['pos_2d'].to(device).unsqueeze(1)  # Add fiducial dimension
                
                pred_2d, _ = model(pos_3d)
                loss = F.mse_loss(pred_2d, pos_2d)
                val_losses.append(loss.item())
        
        model.train()
        
        # Record losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['val_loss'].append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
    
    return model, loss_history


def test_equivariance(model, dataset, device="cpu"):
    """Test if the model is actually equivariant"""
    model.eval()
    
    # Get a sample
    sample = dataset[0]
    pos_3d = sample['pos_3d'].unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, 3]
    
    # Original prediction
    with torch.no_grad():
        pred_original, _ = model(pos_3d)
    
    # Rotate input
    rotation_matrix = o3.rand_matrix()
    pos_3d_rotated = (rotation_matrix @ pos_3d.squeeze().T).T.unsqueeze(0).unsqueeze(1)
    
    # Prediction on rotated input
    with torch.no_grad():
        pred_rotated, _ = model(pos_3d_rotated)
    
    # Rotate the original prediction
    pred_original_rotated = (rotation_matrix[:2, :2] @ pred_original.squeeze().T).T
    
    # Check if they're close
    diff = torch.norm(pred_rotated.squeeze() - pred_original_rotated)
    print(f"Equivariance test difference: {diff.item():.6f}")
    print(f"Equivariant: {diff.item() < 1e-3}")
    
    return diff.item() < 1e-3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Train model
    try:
        model, loss_history = train_simple_e3nn(
            data_dir="data",
            num_epochs=50,
            learning_rate=0.001,
            batch_size=32,
            device="cpu"  # Start with CPU for debugging
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['train_loss'], label='Train Loss')
        plt.plot(loss_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.title('Training Progress')
        plt.show()
        
        # Test equivariance
        parser = ThreeDCTDataParser("data")
        sessions = parser.load_multiple_sessions()
        dataset = FixedFiducialDataset(sessions)
        
        print("\nTesting equivariance...")
        is_equivariant = test_equivariance(model, dataset)
        
        # Save model
        torch.save(model.state_dict(), 'simple_e3nn_model.pth')
        print("Model saved successfully")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()