#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
import traceback

from e3nn_correlator_network import (
    E3NNFiducialCorrelator,
    train_e3nn_correlator,
    TrainingConfig,
    FiducialDataset
)
from fung_data_parser import ThreeDCTDataParser
from synthetic_data_generator import ThreeDCTSyntheticGenerator

def setup_logging(log_dir: str) -> None:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging to {log_file}")

def create_dataset(
    data_dir: str,
    synthetic_ratio: float = 0.7,
    num_synthetic_sessions: int = 10
) -> Tuple[FiducialDataset, Dict]:
    """Create dataset with real and synthetic data"""
    # Load real data
    parser = ThreeDCTDataParser(data_dir)
    real_sessions = parser.load_multiple_sessions()
    logging.info(f"Loaded {len(real_sessions)} real sessions")
    
    # Generate synthetic data
    generator = ThreeDCTSyntheticGenerator.from_real_data(data_dir)
    synthetic_sessions = generator.generate_sessions(num_synthetic_sessions)
    logging.info(f"Generated {len(synthetic_sessions)} synthetic sessions")
    
    # Combine sessions
    all_sessions = real_sessions + synthetic_sessions
    
    # Create dataset
    dataset = FiducialDataset(all_sessions)
    logging.info(f"Created dataset with {len(dataset)} training pairs")
    
    # Get normalization statistics
    stats = {
        'mean_3d': dataset.mean_3d.tolist(),
        'std_3d': dataset.std_3d.tolist(),
        'mean_2d': dataset.mean_2d.tolist(),
        'std_2d': dataset.std_2d.tolist()
    }
    
    return dataset, stats

def create_dataloaders(
    dataset: FiducialDataset,
    batch_size: int,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    # Split dataset
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Created dataloaders:")
    logging.info(f"  - Training set: {train_size} samples")
    logging.info(f"  - Validation set: {val_size} samples")
    
    return train_loader, val_loader

def plot_training_history(
    loss_history: Dict[str, List[float]],
    save_dir: str
) -> None:
    """Plot and save training history"""
    # Create plots directory
    plots_dir = Path(save_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['train_loss'], label='Train Loss')
    plt.plot(loss_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "total_loss.png")
    plt.close()
    
    # Plot component losses
    plt.figure(figsize=(15, 10))
    
    # Rotation loss
    plt.subplot(3, 1, 1)
    plt.plot(loss_history['train_rot_loss'], label='Train')
    plt.plot(loss_history['val_rot_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rotation Loss')
    plt.legend()
    plt.grid(True)
    
    # Scale loss
    plt.subplot(3, 1, 2)
    plt.plot(loss_history['train_scale_loss'], label='Train')
    plt.plot(loss_history['val_scale_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Scale Loss')
    plt.legend()
    plt.grid(True)
    
    # Translation loss
    plt.subplot(3, 1, 3)
    plt.plot(loss_history['train_trans_loss'], label='Train')
    plt.plot(loss_history['val_trans_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Translation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "component_losses.png")
    plt.close()

def save_training_config(
    config: TrainingConfig,
    dataset_stats: Dict,
    save_dir: str
) -> None:
    """Save training configuration and dataset statistics"""
    config_dict = {
        'data_dir': config.data_dir,
        'synthetic_ratio': config.synthetic_ratio,
        'num_synthetic_sessions': config.num_synthetic_sessions,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'device': config.device,
        'dataset_stats': dataset_stats
    }
    
    with open(Path(save_dir) / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)

def main():
    """Main training function"""
    try:
        # Create output directory
        output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(str(output_dir))
        logging.info("Starting training")
        
        # Create training configuration
        config = TrainingConfig(
            data_dir="data",
            synthetic_ratio=0.7,
            num_synthetic_sessions=10,
            num_epochs=100,
            learning_rate=0.001,
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            log_dir=str(output_dir)
        )
        
        # Create dataset
        dataset, dataset_stats = create_dataset(
            config.data_dir,
            config.synthetic_ratio,
            config.num_synthetic_sessions
        )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset,
            config.batch_size
        )
        
        # Train model
        model, loss_history = train_e3nn_correlator(
            config,
            train_loader,
            val_loader
        )
        
        # Plot training history
        plot_training_history(loss_history, str(output_dir))
        
        # Save training configuration
        save_training_config(config, dataset_stats, str(output_dir))
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        logging.error("Stack trace:")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 