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
from torch.utils.data import DataLoader

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
class E3NNFiducialCorrelator(nn.Module):
    """E3NN-based network for learning 3D to 2D fiducial correlation"""
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Input representation: 3D coordinates (x,y,z) for each fiducial
        self.input_irreps = o3.Irreps("3x0e")  # 3 scalar values (x,y,z)
        self.input_dim = self.input_irreps.dim  # Should be 3
        
        # Hidden layer representations
        self.hidden_irreps = o3.Irreps("16x0e")  # Simplified to just scalar representations
        self.hidden_dim = self.hidden_irreps.dim  # Should be 16
        
        # Output representations:
        # - 2D coordinates (x,y) for each fiducial (scalar)
        # - Transformation parameters (rotation matrix, scale, translation)
        self.output_irreps = o3.Irreps("2x0e")  # 2 scalar values (x,y)
        self.output_dim = self.output_irreps.dim  # Should be 2
        
        self.transform_irreps = o3.Irreps("13x0e")  # 13 scalar values (9 rot + 1 scale + 3 trans)
        self.transform_dim = self.transform_irreps.dim  # Should be 13
        
        # Network layers
        self.layers = nn.ModuleList([
            # First layer: 3D -> hidden
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer: hidden -> hidden
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third layer: hidden -> output
            nn.Linear(self.hidden_dim, self.output_dim)
        ])
        
        # Transformation prediction layers
        self.transform_layers = nn.ModuleList([
            # First layer: 3D -> hidden
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer: hidden -> hidden
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third layer: hidden -> transform params
            nn.Linear(self.hidden_dim, self.transform_dim)
        ])
        
    def forward(self, x):
        # Input shape: (batch_size, num_fiducials, 3)
        batch_size = x.shape[0]
        num_fiducials = x.shape[1]
        
        # Reshape input for processing
        x = x.reshape(-1, 3)  # (batch_size * num_fiducials, 3)
        
        # Process through coordinate prediction layers
        coord_features = x
        for layer in self.layers:
            coord_features = layer(coord_features)
        
        # Process through transform prediction layers
        transform_features = x
        for layer in self.transform_layers:
            transform_features = layer(transform_features)
        
        # Reshape back to batch format
        coords_2d = coord_features.reshape(batch_size, num_fiducials, 2)
        transform_params = transform_features.reshape(batch_size, num_fiducials, 13)
        
        return coords_2d, transform_params
    
    def predict_transform(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        rotation = output[:, :, :9].reshape(-1, 3, 3)
        scale = output[:, :, 9:10]
        translation = output[:, :, 10:13]
        center = output[:, :, 13:16]
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
        
        return {
            'x_3d': x,
            'y_2d': y_coords,
            'transform_params': y_transform
        }

def train_e3nn_correlator(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Tuple[E3NNFiducialCorrelator, Dict[str, List[float]]]:
    """Train the E3NN correlator network"""
    # Initialize model
    model = E3NNFiducialCorrelator(dropout_rate=0.3).to(config.device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Initialize loss history
    loss_history = {
        'train_loss': [],
        'val_loss': [],
        'train_rot_loss': [],
        'val_rot_loss': [],
        'train_scale_loss': [],
        'val_scale_loss': [],
        'train_trans_loss': [],
        'val_trans_loss': []
    }
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_rot_losses = []
        train_scale_losses = []
        train_trans_losses = []
        
        for batch in train_loader:
            # Get batch data
            x_3d = batch['x_3d'].to(config.device)
            y_2d = batch['y_2d'].to(config.device)
            transform_params = batch['transform_params'].to(config.device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_2d, pred_transform = model(x_3d)
            
            # Calculate losses
            coord_loss = F.mse_loss(pred_2d, y_2d)
            rot_loss = F.mse_loss(pred_transform[:, :9], transform_params[:, :9])
            scale_loss = F.mse_loss(pred_transform[:, 9], transform_params[:, 9])
            trans_loss = F.mse_loss(pred_transform[:, 10:], transform_params[:, 10:])
            
            total_loss = coord_loss + rot_loss + scale_loss + trans_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            train_losses.append(total_loss.item())
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
                pred_2d, pred_transform = model(x_3d)
                
                # Calculate losses
                coord_loss = F.mse_loss(pred_2d, y_2d)
                rot_loss = F.mse_loss(pred_transform[:, :9], transform_params[:, :9])
                scale_loss = F.mse_loss(pred_transform[:, 9], transform_params[:, 9])
                trans_loss = F.mse_loss(pred_transform[:, 10:], transform_params[:, 10:])
                
                total_loss = coord_loss + rot_loss + scale_loss + trans_loss
                
                # Record losses
                val_losses.append(total_loss.item())
                val_rot_losses.append(rot_loss.item())
                val_scale_losses.append(scale_loss.item())
                val_trans_losses.append(trans_loss.item())
        
        # Update learning rate
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Record epoch losses
        loss_history['train_loss'].append(np.mean(train_losses))
        loss_history['val_loss'].append(avg_val_loss)
        loss_history['train_rot_loss'].append(np.mean(train_rot_losses))
        loss_history['val_rot_loss'].append(np.mean(val_rot_losses))
        loss_history['train_scale_loss'].append(np.mean(train_scale_losses))
        loss_history['val_scale_loss'].append(np.mean(val_scale_losses))
        loss_history['train_trans_loss'].append(np.mean(train_trans_losses))
        loss_history['val_trans_loss'].append(np.mean(val_trans_losses))
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{config.num_epochs}")
        logging.info(f"Train Loss: {loss_history['train_loss'][-1]:.4f}")
        logging.info(f"Val Loss: {loss_history['val_loss'][-1]:.4f}")
        logging.info(f"Train Rot Loss: {loss_history['train_rot_loss'][-1]:.4f}")
        logging.info(f"Val Rot Loss: {loss_history['val_rot_loss'][-1]:.4f}")
        logging.info(f"Train Scale Loss: {loss_history['train_scale_loss'][-1]:.4f}")
        logging.info(f"Val Scale Loss: {loss_history['val_scale_loss'][-1]:.4f}")
        logging.info(f"Train Trans Loss: {loss_history['train_trans_loss'][-1]:.4f}")
        logging.info(f"Val Trans Loss: {loss_history['val_trans_loss'][-1]:.4f}")
    
    return model, loss_history

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