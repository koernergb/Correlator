#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
from typing import Tuple, List

# Import directly from root directory
from fung_data_parser import ThreeDCTDataParser, CorrelationSession
from synthetic_data_generator import ThreeDCTSyntheticGenerator
from e3nn_correlator_network import E3NNFiducialCorrelator

def split_session(session: CorrelationSession, val_ratio: float = 0.3) -> Tuple[CorrelationSession, CorrelationSession]:
    """Split a single correlation session into training and validation sets"""
    print("\nSplitting real session into train/val sets...")
    
    # Get all fiducial pairs
    all_pairs = session.fiducial_pairs
    n_pairs = len(all_pairs)
    n_val = int(n_pairs * val_ratio)
    
    # Randomly select validation pairs
    indices = np.random.permutation(n_pairs)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Create validation session
    val_session = CorrelationSession(
        session_id=f"{session.session_id}_val",
        machine_info=session.machine_info,
        date=session.date,
        transformation=session.transformation,
        fiducial_pairs=[all_pairs[i] for i in val_indices],
        pois=session.pois,  # Keep all POIs in both sets
        csv_coordinates=None,
        microscope_center=session.microscope_center,
        pixel_size_um=session.pixel_size_um
    )
    
    # Create training session
    train_session = CorrelationSession(
        session_id=f"{session.session_id}_train",
        machine_info=session.machine_info,
        date=session.date,
        transformation=session.transformation,
        fiducial_pairs=[all_pairs[i] for i in train_indices],
        pois=session.pois,  # Keep all POIs in both sets
        csv_coordinates=None,
        microscope_center=session.microscope_center,
        pixel_size_um=session.pixel_size_um
    )
    
    print(f"Split {n_pairs} fiducials into:")
    print(f"  - Training: {len(train_session.fiducial_pairs)} fiducials")
    print(f"  - Validation: {len(val_session.fiducial_pairs)} fiducials")
    
    return train_session, val_session

def load_trained_model(model_path: str, device: str = "cpu") -> E3NNFiducialCorrelator:
    """Load a trained model from disk"""
    print(f"\nLoading model from: {model_path}")
    
    # Initialize model with same architecture
    model = E3NNFiducialCorrelator(
        irreps_in="1x0e + 1x1e",
        irreps_hidden="16x0e + 16x1e + 16x2e",
        irreps_out="1x0e + 1x1e"
    )
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model

def generate_synthetic_data(train_session: CorrelationSession, n_sessions: int = 5, seed: int = 42) -> List[CorrelationSession]:
    """Generate synthetic data using only the training portion of real data"""
    print(f"\nGenerating {n_sessions} synthetic sessions from training data...")
    
    # Create generator using only training data
    generator = ThreeDCTSyntheticGenerator.from_real_data("data/2023_embo_clem_material/3DCT/data")
    
    # Generate synthetic sessions
    synthetic_sessions = generator.generate_sessions(n_sessions, seed=seed)
    print(f"✓ Generated {len(synthetic_sessions)} synthetic sessions")
    
    return synthetic_sessions

def evaluate_predictions(
    model: E3NNFiducialCorrelator,
    test_sessions: List[CorrelationSession],
    device: str = "cpu"
) -> dict:
    """Evaluate model predictions on test data"""
    print("\nEvaluating model predictions...")
    
    all_errors = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for session in test_sessions:
            # Extract test pairs
            parser = ThreeDCTDataParser()
            test_pairs = parser.extract_training_pairs(session)
            
            # Convert to tensors
            x = torch.tensor([pair[0] for pair in test_pairs], dtype=torch.float32).to(device)
            y_true = torch.tensor([pair[1] for pair in test_pairs], dtype=torch.float32).to(device)
            
            # Make predictions
            y_pred = model(x)
            
            # Calculate errors
            errors = torch.norm(y_pred - y_true, dim=1)
            all_errors.extend(errors.cpu().numpy())
            all_predictions.extend(y_pred.cpu().numpy())
            all_targets.extend(y_true.cpu().numpy())
    
    # Convert to numpy arrays
    all_errors = np.array(all_errors)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate statistics
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    max_error = np.max(all_errors)
    min_error = np.min(all_errors)
    
    print("\nPrediction Statistics:")
    print(f"Mean error: {mean_error:.2f} pixels")
    print(f"Std error: {std_error:.2f} pixels")
    print(f"Max error: {max_error:.2f} pixels")
    print(f"Min error: {min_error:.2f} pixels")
    
    return {
        'errors': all_errors,
        'predictions': all_predictions,
        'targets': all_targets,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'min_error': min_error
    }

def visualize_results(results: dict, title: str = "Model Predictions"):
    """Visualize prediction results"""
    print("\nVisualizing results...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    
    # 1. Error distribution
    ax1 = fig.add_subplot(131)
    ax1.hist(results['errors'], bins=20, alpha=0.7)
    ax1.set_xlabel('Error (pixels)')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    
    # 2. Scatter plot of predictions vs targets
    ax2 = fig.add_subplot(132)
    ax2.scatter(results['targets'][:, 0], results['predictions'][:, 0], 
                alpha=0.5, label='X coordinate')
    ax2.scatter(results['targets'][:, 1], results['predictions'][:, 1], 
                alpha=0.5, label='Y coordinate')
    ax2.plot([0, 2000], [0, 2000], 'k--', alpha=0.3)  # Diagonal line
    ax2.set_xlabel('Target (pixels)')
    ax2.set_ylabel('Prediction (pixels)')
    ax2.set_title('Predictions vs Targets')
    ax2.legend()
    
    # 3. Error magnitude vs target position
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(results['targets'][:, 0], results['targets'][:, 1],
                         c=results['errors'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Error (pixels)')
    ax3.set_xlabel('Target X (pixels)')
    ax3.set_ylabel('Target Y (pixels)')
    ax3.set_title('Error Magnitude by Position')
    
    # Add statistics text
    stats_text = f"""Prediction Statistics:
Mean error: {results['mean_error']:.2f} pixels
Std error: {results['std_error']:.2f} pixels
Max error: {results['max_error']:.2f} pixels
Min error: {results['min_error']:.2f} pixels"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.show()

def main():
    print("\n=== Testing E3NN Correlator Model ===")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load latest model
    model_dir = Path("models")
    model_files = list(model_dir.glob("e3nn_correlator_*.pth"))
    if not model_files:
        print("✗ No trained models found in models/ directory")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Found latest model: {latest_model.name}")
    
    # Load model
    model = load_trained_model(str(latest_model), device)
    
    # Load real data
    print("\nLoading real data...")
    parser = ThreeDCTDataParser("data/2023_embo_clem_material/3DCT/data")
    real_sessions = parser.load_multiple_sessions()
    
    if not real_sessions:
        print("✗ No real sessions found")
        return
    
    # Split the real session into train/val
    train_session, val_session = split_session(real_sessions[0])
    
    # Generate synthetic data using only training portion
    synthetic_sessions = generate_synthetic_data(train_session, n_sessions=5, seed=42)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_predictions(model, [val_session], device)
    visualize_results(val_results, "Validation Set Performance")
    
    # Evaluate on synthetic data
    print("\nEvaluating on synthetic data...")
    synth_results = evaluate_predictions(model, synthetic_sessions, device)
    visualize_results(synth_results, "Synthetic Data Performance")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1) 