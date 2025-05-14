#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging
import torch
from datetime import datetime

# Import directly from root directory
from fung_data_parser import ThreeDCTDataParser
from synthetic_data_generator import ThreeDCTSyntheticGenerator
from e3nn_correlator_network import train_e3nn_correlator, E3NNFiducialCorrelator

def setup_directories():
    """Create necessary directories if they don't exist"""
    print("\n=== Setting up Project Directories ===")
    dirs = [
        "data/2023_embo_clem_material/3DCT/data",
        "logs",
        "models"
    ]
    for dir_path in dirs:
        print(f"Creating directory: {dir_path}")
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created/verified: {dir_path}")

def main():
    print("\n=== Starting E3NN Correlator Training ===")
    
    # Setup directories
    setup_directories()
    
    # Training parameters
    data_dir = "data/2023_embo_clem_material/3DCT/data"
    synthetic_ratio = 0.7  # 70% synthetic, 30% real data
    n_synthetic_sessions = 10
    num_epochs = 100
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n=== Training Configuration ===")
    print(f"Data directory: {data_dir}")
    print(f"Synthetic ratio: {synthetic_ratio}")
    print(f"Number of synthetic sessions: {n_synthetic_sessions}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nRun timestamp: {timestamp}")
    
    # Setup logging
    log_dir = f"logs/training_{timestamp}"
    print(f"\nSetting up logging in: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    print("✓ Log directory created")
    
    # Train the model
    print("\n=== Starting Model Training ===")
    try:
        print("\nInitializing training...")
        model, loss_history = train_e3nn_correlator(
            data_dir=data_dir,
            synthetic_ratio=synthetic_ratio,
            n_synthetic_sessions=n_synthetic_sessions,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            log_dir=log_dir
        )
        print("✓ Training completed successfully")
        
        # Save the model
        print("\n=== Saving Model ===")
        model_path = f"models/e3nn_correlator_{timestamp}.pth"
        print(f"Saving model to: {model_path}")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'training_params': {
                'synthetic_ratio': synthetic_ratio,
                'n_synthetic_sessions': n_synthetic_sessions,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate
            }
        }, model_path)
        
        print("✓ Model saved successfully")
        logging.info(f"Model saved to {model_path}")
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Best loss: {min(loss_history):.6f}")
        print(f"Training log: {log_dir}")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Program failed with error: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1) 