#!/usr/bin/env python3
# src/models/e3nn_correlator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import e3nn
from e3nn import o3, nn as e3nn_nn
from e3nn.o3 import Irreps
from e3nn.nn import FullyConnectedNet
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from datetime import datetime
import traceback
import random
import os
from dataclasses import dataclass
from e3nn.util.jit import compile_mode
import json

# Import directly from root directory
from fung_data_parser import ThreeDCTDataParser, CorrelationSession, TransformationParams
from synthetic_data_generator import ThreeDCTSyntheticGenerator

@dataclass
class TrainingConfig:
    """Configuration for training"""
    data_dir: str
    synthetic_ratio: float = 0.7
    num_synthetic_sessions: int = 10
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Optional[str] = None

@compile_mode("script")
class E3NNFiducialCorrelator(torch.nn.Module):
    """E3NN-based network for learning 3D to 2D fiducial correlation"""
    
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define irreducible representations for input and output
        self.irreps_in = Irreps("1x1o")  # 3D coordinates as scalar
        self.irreps_out = Irreps("1x1o")  # 2D coordinates as scalar
        
        # Define hidden layer irreps with higher order spherical harmonics
        self.irreps_hidden = Irreps("16x0e + 16x1o + 16x2e")  # More channels and higher order
        
        # Input normalization layer
        self.input_norm = torch.nn.BatchNorm1d(3)
        
        # E3NN layers with proper initialization
        self.layers = torch.nn.ModuleList([
            # First layer: 3D to hidden representation
            e3nn_nn.FullyConnectedNet(
                [self.irreps_in, self.irreps_hidden],
                act=torch.nn.ReLU()
            ),
            
            # Second layer: Process geometric features
            e3nn_nn.FullyConnectedNet(
                [self.irreps_hidden, self.irreps_hidden],
                act=torch.nn.ReLU()
            ),
            
            # Third layer: Process geometric relationships
            e3nn_nn.FullyConnectedNet(
                [self.irreps_hidden, self.irreps_hidden],
                act=torch.nn.ReLU()
            ),
            
            # Final layer: Project to 2D
            e3nn_nn.FullyConnectedNet(
                [self.irreps_hidden, self.irreps_out],
                act=None  # No activation for final layer
            )
        ])
        
        # Add dropout layers
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        
        # Initialize weights with better scaling
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights with better scaling for geometric relationships"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            # Use Kaiming initialization with better scaling
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved geometric processing"""
        # Input normalization
        x = self.input_norm(x)
        
        # Process through E3NN layers with dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply dropout after each layer except the last one
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        
        # Extract 2D coordinates (first two components)
        return x[:, :2]
    
    def predict_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Predict transformation parameters"""
        # Get the full output from the network
        output = self.forward(x)
        
        # Extract transformation parameters
        # First 9 elements: rotation matrix
        rotation = output[:, :9].reshape(-1, 3, 3)
        # Next element: scale
        scale = output[:, 9:10]
        # Next 3 elements: translation
        translation = output[:, 10:13]
        # Last 3 elements: center point
        center = output[:, 13:16]
        
        return torch.cat([
            rotation.reshape(-1, 9),
            scale,
            translation,
            center
        ], dim=1)

class GeometricLoss(torch.nn.Module):
    """Custom loss function for geometric transformation parameters"""
    
    def __init__(self):
        super().__init__()
        
        # Weights for different components
        self.rotation_weight = 1.0
        self.scale_weight = 0.1
        self.translation_weight = 0.01
        self.center_weight = 0.01
        
        # Scale factors for translation and center point
        self.trans_scale = 0.001
        self.center_scale = 0.001
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between predicted and target transformation parameters
        
        Args:
            pred: Predicted transformation parameters [batch_size, 16]
            target: Target transformation parameters [batch_size, 16]
            
        Returns:
            Combined loss value
        """
        # Split parameters into components
        pred_rot = pred[:, :9].view(-1, 3, 3)  # Rotation matrix
        pred_scale = pred[:, 9]  # Scale
        pred_trans = pred[:, 10:13]  # Translation
        pred_center = pred[:, 13:]  # Center point
        
        target_rot = target[:, :9].view(-1, 3, 3)
        target_scale = target[:, 9]
        target_trans = target[:, 10:13]
        target_center = target[:, 13:]
        
        # Rotation matrix loss (Frobenius norm)
        rot_loss = torch.mean(torch.norm(pred_rot - target_rot, dim=(1, 2)))
        
        # Scale loss (relative error)
        scale_loss = torch.mean(torch.abs(pred_scale - target_scale) / target_scale)
        
        # Translation loss (MSE)
        trans_loss = torch.mean(torch.sum((pred_trans - target_trans) ** 2, dim=1))
        
        # Center point loss (MSE)
        center_loss = torch.mean(torch.sum((pred_center - target_center) ** 2, dim=1))
        
        # Rotation matrix constraint: R^T R = I
        pred_rot_t = pred_rot.transpose(1, 2)
        rot_constraint = torch.mean(torch.norm(
            torch.bmm(pred_rot_t, pred_rot) - torch.eye(3, device=pred.device).unsqueeze(0),
            dim=(1, 2)
        ))
        
        # Scale constraint: ensure positive scale
        scale_constraint = torch.mean(torch.relu(-pred_scale))
        
        # Combine losses with weights
        total_loss = (
            self.rotation_weight * (rot_loss + 0.1 * rot_constraint) +
            self.scale_weight * (scale_loss + 0.1 * scale_constraint) +
            self.translation_weight * trans_loss +
            self.center_weight * center_loss
        )
        
        return total_loss

class FiducialDataset(torch.utils.data.Dataset):
    """Dataset for fiducial marker correlation"""
    
    def __init__(self, sessions: List[CorrelationSession]):
        self.sessions = sessions
        self.pairs = []
        
        # Extract training pairs and transformation parameters from each session
        for session in sessions:
            # Get transformation parameters
            transform = session.transformation
            
            # Convert transformation parameters to E3NN format
            transform_params = self._convert_to_e3nn_format(transform)
            
            # Get fiducial pairs
            pairs = ThreeDCTDataParser().extract_training_pairs(session)
            
            # Store pairs with their transformation parameters
            for pair in pairs:
                self.pairs.append((pair[0], pair[1], transform_params))
        
        # Calculate normalization statistics
        self._calculate_normalization_stats()
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization"""
        all_3d_coords = np.array([pair[0] for pair in self.pairs])
        all_2d_coords = np.array([pair[1] for pair in self.pairs])
        
        # Calculate statistics for 3D coordinates
        self.mean_3d = np.mean(all_3d_coords, axis=0)
        self.std_3d = np.std(all_3d_coords, axis=0)
        self.std_3d[self.std_3d == 0] = 1.0  # Prevent division by zero
        
        # Calculate statistics for 2D coordinates
        self.mean_2d = np.mean(all_2d_coords, axis=0)
        self.std_2d = np.std(all_2d_coords, axis=0)
        self.std_2d[self.std_2d == 0] = 1.0  # Prevent division by zero
    
    def _normalize_3d(self, coords: np.ndarray) -> np.ndarray:
        """Normalize 3D coordinates"""
        return (coords - self.mean_3d) / self.std_3d
    
    def _normalize_2d(self, coords: np.ndarray) -> np.ndarray:
        """Normalize 2D coordinates"""
        return (coords - self.mean_2d) / self.std_2d
    
    def _denormalize_2d(self, coords: np.ndarray) -> np.ndarray:
        """Denormalize 2D coordinates"""
        return coords * self.std_2d + self.mean_2d
    
    def _convert_to_e3nn_format(self, transform: TransformationParams) -> np.ndarray:
        """Convert transformation parameters to E3NN format"""
        # Convert Euler angles to rotation matrix
        phi, psi, theta = np.radians(transform.rotation_euler)
        
        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R_y = np.array([
            [np.cos(psi), 0, np.sin(psi)],
            [0, 1, 0],
            [-np.sin(psi), 0, np.cos(psi)]
        ])
        
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = R_z @ R_y @ R_x
        
        # Flatten rotation matrix and combine with other parameters
        params = np.concatenate([
            R.flatten(),  # 9 parameters for rotation
            [transform.scale],  # 1 parameter for scale
            transform.translation_center,  # 3 parameters for translation
            transform.center_point  # 3 parameters for center point
        ])
        
        return params
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Create input features and targets with normalization
        x = torch.tensor(self._normalize_3d(pair[0]), dtype=torch.float32)  # Normalized 3D coordinates
        y_coords = torch.tensor(self._normalize_2d(pair[1]), dtype=torch.float32)  # Normalized 2D coordinates
        y_transform = torch.tensor(pair[2], dtype=torch.float32)  # Transformation parameters
        
        return x, y_coords, y_transform

def train_e3nn_correlator(
    config: TrainingConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: Optional[E3NNFiducialCorrelator] = None
) -> Tuple[E3NNFiducialCorrelator, Dict[str, List[float]]]:
    """Train the E3NN correlator model with improved training loop"""
    try:
        # Initialize model if not provided
        if model is None:
            model = E3NNFiducialCorrelator(dropout_rate=0.3).to(config.device)
        
        # Initialize optimizer with better learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize loss history
        loss_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rot_loss': [],
            'train_scale_loss': [],
            'train_trans_loss': [],
            'val_rot_loss': [],
            'val_scale_loss': [],
            'val_trans_loss': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_rot_losses = []
            train_scale_losses = []
            train_trans_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Get batch data
                x_3d = batch['x_3d'].to(config.device)
                y_2d = batch['y_2d'].to(config.device)
                transform_params = batch['transform_params'].to(config.device)
                
                # Forward pass
                pred_2d = model(x_3d)
                
                # Calculate losses
                coord_loss = F.mse_loss(pred_2d, y_2d)
                
                # Get predicted transformation parameters
                pred_params = model.predict_transform(x_3d)
                
                # Calculate transformation losses
                rot_loss = F.mse_loss(pred_params[:, :9], transform_params[:, :9])
                scale_loss = F.mse_loss(pred_params[:, 9:10], transform_params[:, 9:10])
                trans_loss = F.mse_loss(pred_params[:, 10:13], transform_params[:, 10:13])
                
                # Combined loss
                loss = coord_loss + 0.1 * (rot_loss + scale_loss + trans_loss)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Record losses
                train_losses.append(loss.item())
                train_rot_losses.append(rot_loss.item())
                train_scale_losses.append(scale_loss.item())
                train_trans_losses.append(trans_loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            val_rot_losses = []
            val_scale_losses = []
            val_trans_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Get batch data
                    x_3d = batch['x_3d'].to(config.device)
                    y_2d = batch['y_2d'].to(config.device)
                    transform_params = batch['transform_params'].to(config.device)
                    
                    # Forward pass
                    pred_2d = model(x_3d)
                    
                    # Calculate losses
                    coord_loss = F.mse_loss(pred_2d, y_2d)
                    
                    # Get predicted transformation parameters
                    pred_params = model.predict_transform(x_3d)
                    
                    # Calculate transformation losses
                    rot_loss = F.mse_loss(pred_params[:, :9], transform_params[:, :9])
                    scale_loss = F.mse_loss(pred_params[:, 9:10], transform_params[:, 9:10])
                    trans_loss = F.mse_loss(pred_params[:, 10:13], transform_params[:, 10:13])
                    
                    # Combined loss
                    loss = coord_loss + 0.1 * (rot_loss + scale_loss + trans_loss)
                    
                    # Record losses
                    val_losses.append(loss.item())
                    val_rot_losses.append(rot_loss.item())
                    val_scale_losses.append(scale_loss.item())
                    val_trans_losses.append(trans_loss.item())
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Record losses
            loss_history['train_loss'].append(avg_train_loss)
            loss_history['val_loss'].append(avg_val_loss)
            loss_history['train_rot_loss'].append(np.mean(train_rot_losses))
            loss_history['train_scale_loss'].append(np.mean(train_scale_losses))
            loss_history['train_trans_loss'].append(np.mean(train_trans_losses))
            loss_history['val_rot_loss'].append(np.mean(val_rot_losses))
            loss_history['val_scale_loss'].append(np.mean(val_scale_losses))
            loss_history['val_trans_loss'].append(np.mean(val_trans_losses))
            
            # Print progress
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {avg_val_loss:.6f}")
            print(f"Train Rot Loss: {np.mean(train_rot_losses):.6f}")
            print(f"Train Scale Loss: {np.mean(train_scale_losses):.6f}")
            print(f"Train Trans Loss: {np.mean(train_trans_losses):.6f}")
            print(f"Val Rot Loss: {np.mean(val_rot_losses):.6f}")
            print(f"Val Scale Loss: {np.mean(val_scale_losses):.6f}")
            print(f"Val Trans Loss: {np.mean(val_trans_losses):.6f}")
            print("-" * 50)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                if config.log_dir:
                    torch.save(model.state_dict(), Path(config.log_dir) / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return model, loss_history
    
    except Exception as e:
        print(f"Error in training: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Test the model
    data_dir = "data/2023_embo_clem_material/3DCT/data"
    model, loss_history = train_e3nn_correlator(
        data_dir=data_dir,
        synthetic_ratio=0.7,
        num_synthetic_sessions=10,
        num_epochs=100,
        learning_rate=0.001
    )