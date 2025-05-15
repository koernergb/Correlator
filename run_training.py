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
import argparse

from better_correlator import (
    SimpleE3NNCorrelator,
    train_simple_e3nn,
    FixedFiducialDataset
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

def main():
    """Main training function"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train E3NN correlator network')
        parser.add_argument('--num_epochs', type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                          help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size')
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                          help='Device to use (cuda/cpu/mps)')
        args = parser.parse_args()

        # Create output directory
        output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(str(output_dir))
        logging.info("Starting training")
        
        # Load real data
        parser = ThreeDCTDataParser("data/2023_embo_clem_material/3DCT/data")
        real_sessions = parser.load_multiple_sessions()

        # Generate synthetic data
        generator = ThreeDCTSyntheticGenerator.from_real_data("data/2023_embo_clem_material/3DCT/data")
        num_synthetic_sessions = 1000  # or any number you want
        synthetic_sessions = generator.generate_sessions(num_synthetic_sessions)

        # Combine sessions
        all_sessions = real_sessions + synthetic_sessions

        # Create dataset
        dataset = FixedFiducialDataset(all_sessions)
        
        # Train model
        model, loss_history = train_simple_e3nn(
            data_dir="data/2023_embo_clem_material/3DCT/data",
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history['train_loss'], label='Train Loss')
        plt.plot(loss_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.title('Training Progress')
        plt.savefig(output_dir / "training_history.png")
        
        # Save model
        torch.save(model.state_dict(), output_dir / "model.pth")
        logging.info("Model saved successfully")
        
        # Save training config
        config = {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'device': args.device
        }
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        logging.error("Stack trace:")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 