# src/models/e3nn_correlator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import e3nn
from e3nn import o3, nn as e3nn_nn
from e3nn.nn import BatchNorm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_analysis.tdct_data_parser import ThreeDCTDataParser, CorrelationSession

def setup_logging(log_dir: str = "logs") -> str:
    """Setup logging to both file and console
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"e3nn_training_{timestamp}.log"
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")
    return str(log_file)

class E3NNFiducialCorrelator(nn.Module):
    """E3NN-based network for 3D→2D fiducial correlation in 3DCT"""
    
    def __init__(self, 
                 max_degree: int = 2,
                 hidden_features: int = 64,
                 num_layers: int = 3,
                 max_radius: float = 5.0,
                 num_neighbors: int = 8):
        """
        Initialize E3NN correlator
        
        Args:
            max_degree: Maximum spherical harmonic degree
            hidden_features: Number of features in hidden layers
            num_layers: Number of interaction layers
            max_radius: Maximum radius for neighbor finding
            num_neighbors: Number of neighbors for graph construction
        """
        super().__init__()
        
        self.max_degree = max_degree
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        
        # Define irreducible representations (irreps)
        # Start with just scalars for positions
        irreps_in = o3.Irreps("1x0e")  # Input: 3D positions (scalar)
        irreps_hidden = self._build_irreps(hidden_features, max_degree)
        irreps_out = o3.Irreps("1x0e")  # Output: scalars that will become 2D coords
        
        # Initial embedding
        self.embedding = e3nn_nn.Linear(irreps_in, irreps_hidden)
        
        # E3NN convolution layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Convolution layer
            layer = TensorProductConvolution(
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                max_degree=max_degree,
                num_neighbors=num_neighbors
            )
            self.layers.append(layer)
            
            # Batch normalization
            self.batch_norms.append(BatchNorm(irreps_hidden))
        
        # Final projection to 2D coordinates
        self.final_projection = nn.Sequential(
            e3nn_nn.Linear(irreps_hidden, o3.Irreps("32x0e")),  # To scalars
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output 2D coordinates
        )
        
    def _build_irreps(self, num_features: int, max_degree: int) -> o3.Irreps:
        """Build irreps for hidden layers with different polynomial degrees"""
        irreps = []
        
        # Distribute features across different degrees
        for degree in range(max_degree + 1):
            if degree == 0:
                # More scalars
                irreps.append(f"{num_features//2}x{degree}e")
            else:
                # Fewer higher-order features
                n_features = max(1, num_features // (4 * degree))
                irreps.append(f"{n_features}x{degree}e")
        
        return o3.Irreps(" + ".join(irreps))
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            positions: 3D positions [batch_size, num_points, 3]
            
        Returns:
            2D coordinates [batch_size, num_points, 2]
        """
        batch_size, num_points, _ = positions.shape
        
        # Build graph based on distances
        edge_src, edge_dst = self._build_graph(positions)
        
        # Initial node features (just use constant features for simplicity)
        # In reality, you might want to add more sophisticated node features
        node_features = torch.ones(batch_size * num_points, 1, device=positions.device)
        
        # Flatten positions for processing
        positions_flat = positions.view(-1, 3)
        
        # Initial embedding
        x = self.embedding(node_features)
        
        # Apply E3NN layers
        for layer, bn in zip(self.layers, self.batch_norms):
            x_new = layer(x, positions_flat, edge_src, edge_dst)
            x_new = bn(x_new)
            x = F.relu(x_new) + x  # Residual connection
        
        # Extract scalar features and project to 2D
        scalar_features = self._extract_scalars(x)
        output_2d = self.final_projection(scalar_features)
        
        # Reshape to batch form
        output_2d = output_2d.view(batch_size, num_points, 2)
        
        return output_2d
    
    def _build_graph(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build k-NN graph from positions"""
        batch_size, num_points, _ = positions.shape
        
        # Compute pairwise distances
        pos_flat = positions.view(-1, 3)
        distances = torch.cdist(pos_flat.unsqueeze(0), pos_flat.unsqueeze(0)).squeeze(0)
        
        # Find k nearest neighbors
        _, indices = torch.topk(distances, self.num_neighbors + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # Remove self-connection
        
        # Create edge indices
        src = torch.arange(num_points * batch_size, device=positions.device).repeat_interleave(self.num_neighbors)
        dst = indices.flatten()
        
        return src, dst
    
    def _extract_scalars(self, features: torch.Tensor) -> torch.Tensor:
        """Extract scalar features from irrep representation"""
        # Find scalar irreps (degree 0)
        scalar_indices = []
        start_idx = 0
        
        for irrep in self.layers[0].irreps_out:
            degree = irrep.ir.l
            dim = irrep.ir.dim
            mul = irrep.mul
            
            if degree == 0:  # Scalar
                scalar_indices.extend(range(start_idx, start_idx + mul * dim))
            start_idx += mul * dim
        
        return features[:, scalar_indices]


class TensorProductConvolution(nn.Module):
    """E3NN tensor product convolution layer"""
    
    def __init__(self, 
                 irreps_in: o3.Irreps,
                 irreps_out: o3.Irreps,
                 max_degree: int,
                 num_neighbors: int):
        super().__init__()
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.max_degree = max_degree
        self.num_neighbors = num_neighbors
        
        # Spherical harmonics for edge attributes
        self.sh = o3.SphericalHarmonics(
            list(range(max_degree + 1)),
            normalize=True,
            normalization='component'
        )
        
        # Tensor product for message generation
        irreps_edge = self.sh.irreps_out
        irreps_mid = []
        instructions = []
        
        # Build tensor product instructions
        for i, (mul_in, ir_in) in enumerate(irreps_in):
            for j, (mul_edge, ir_edge) in enumerate(irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in [ir for _, ir in irreps_out]:
                        irreps_mid.append((mul_in * mul_edge, ir_out))
                        instructions.append((i, j, len(irreps_mid) - 1, 'uvu', True))
        
        irreps_mid = o3.Irreps(irreps_mid)
        
        # Tensor product
        self.tp = o3.TensorProduct(
            irreps_in,
            irreps_edge,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        
        # Linear layer for combining messages
        self.linear = e3nn_nn.Linear(irreps_mid, irreps_out)
        
        # Weights for tensor product
        self.weight = nn.Parameter(torch.randn(self.tp.weight_numel))
        
    def forward(self, 
                node_features: torch.Tensor,
                positions: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor) -> torch.Tensor:
        """Forward pass of tensor product convolution"""
        
        # Compute edge vectors
        edge_vec = positions[edge_dst] - positions[edge_src]
        
        # Compute spherical harmonics for edges
        edge_sh = self.sh(edge_vec)
        
        # Gather source node features
        src_features = node_features[edge_src]
        
        # Apply tensor product
        messages = self.tp(src_features, edge_sh, self.weight)
        
        # Aggregate messages by destination node
        node_out = torch.zeros_like(node_features[:, :self.irreps_out.dim])
        node_out.index_add_(0, edge_dst, messages)
        
        # Apply final linear transformation
        node_out = self.linear(node_out)
        
        return node_out


class GeometricLoss(nn.Module):
    """Geometric loss functions for 3DCT correlation"""
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 distance_weight: float = 0.2,
                 angle_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def forward(self, 
                pred_2d: torch.Tensor,
                target_2d: torch.Tensor,
                points_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute geometric loss
        
        Args:
            pred_2d: Predicted 2D coordinates [batch, num_points, 2]
            target_2d: Target 2D coordinates [batch, num_points, 2]
            points_3d: Original 3D coordinates [batch, num_points, 3]
            
        Returns:
            Dictionary of loss components
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(pred_2d, target_2d)
        
        # Distance preservation loss
        distance_loss = self._distance_preservation_loss(pred_2d, target_2d, points_3d)
        
        # Triangle preservation loss
        angle_loss = self._triangle_preservation_loss(pred_2d, target_2d)
        
        # Total loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.distance_weight * distance_loss + 
                     self.angle_weight * angle_loss)
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'distance': distance_loss,
            'angle': angle_loss
        }
    
    def _distance_preservation_loss(self, 
                                   pred_2d: torch.Tensor,
                                   target_2d: torch.Tensor,
                                   points_3d: torch.Tensor) -> torch.Tensor:
        """Preserve relative distance relationships"""
        # Project 3D to 2D for reference (XY plane)
        reference_2d = points_3d[..., :2]
        
        # Compute distance ratios
        def distance_ratios(points):
            dists = torch.cdist(points, points)
            # Avoid division by zero
            mean_dist = dists.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            return dists / (mean_dist + 1e-8)
        
        ratio_ref = distance_ratios(reference_2d)
        ratio_pred = distance_ratios(pred_2d)
        ratio_target = distance_ratios(target_2d)
        
        return F.mse_loss(ratio_pred, ratio_target)
    
    def _triangle_preservation_loss(self, 
                                   pred_2d: torch.Tensor,
                                   target_2d: torch.Tensor) -> torch.Tensor:
        """Preserve triangular relationships between points"""
        batch_size, num_points, _ = pred_2d.shape
        
        if num_points < 3:
            return torch.tensor(0.0, device=pred_2d.device)
        
        # Sample random triangles
        triangle_loss = 0.0
        num_triangles = min(10, num_points * (num_points - 1) * (num_points - 2) // 6)
        
        for _ in range(num_triangles):
            # Random triangle indices
            indices = torch.randperm(num_points)[:3]
            
            # Get triangle points
            triangle_pred = pred_2d[:, indices, :]
            triangle_target = target_2d[:, indices, :]
            
            # Compute triangle areas
            area_pred = self._triangle_area(triangle_pred)
            area_target = self._triangle_area(triangle_target)
            
            # Area ratio preservation
            ratio_pred = area_pred / (area_pred.sum(dim=-1, keepdim=True) + 1e-8)
            ratio_target = area_target / (area_target.sum(dim=-1, keepdim=True) + 1e-8)
            
            triangle_loss += F.mse_loss(ratio_pred, ratio_target)
        
        return triangle_loss / num_triangles
    
    def _triangle_area(self, triangle_points: torch.Tensor) -> torch.Tensor:
        """Compute area of triangles using cross product"""
        # triangle_points: [batch, 3, 2]
        v1 = triangle_points[:, 1, :] - triangle_points[:, 0, :]
        v2 = triangle_points[:, 2, :] - triangle_points[:, 0, :]
        
        # 2D cross product
        area = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        return area


class ThreeDCTDataset(torch.utils.data.Dataset):
    """Dataset for 3DCT correlation data"""
    
    def __init__(self, 
                 sessions: List[CorrelationSession],
                 parser: Optional[ThreeDCTDataParser] = None,
                 normalize: bool = True):
        self.sessions = sessions
        self.normalize = normalize
        self.parser = parser or ThreeDCTDataParser()
        
        # Extract all training pairs
        self.training_data = []
        for session in sessions:
            training_pairs = self.parser.extract_training_pairs(session)
            for fluor_3d, sem_2d in training_pairs:
                self.training_data.append((fluor_3d, sem_2d))
        
        # Compute normalization statistics
        if normalize:
            all_3d = np.array([pair[0] for pair in self.training_data])
            all_2d = np.array([pair[1] for pair in self.training_data])
            
            self.mean_3d = torch.tensor(np.mean(all_3d, axis=0), dtype=torch.float32)
            self.std_3d = torch.tensor(np.std(all_3d, axis=0), dtype=torch.float32)
            self.mean_2d = torch.tensor(np.mean(all_2d, axis=0), dtype=torch.float32)
            self.std_2d = torch.tensor(np.std(all_2d, axis=0), dtype=torch.float32)
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training pair"""
        fluor_3d, sem_2d = self.training_data[idx]
        
        # Convert to tensors
        positions_3d = torch.tensor(fluor_3d, dtype=torch.float32)
        positions_2d = torch.tensor(sem_2d, dtype=torch.float32)
        
        # Normalize if requested
        if self.normalize:
            positions_3d = (positions_3d - self.mean_3d) / (self.std_3d + 1e-8)
            positions_2d = (positions_2d - self.mean_2d) / (self.std_2d + 1e-8)
        
        return positions_3d, positions_2d


def validate_session_format(session: CorrelationSession) -> bool:
    """Validate that a session matches the expected format"""
    try:
        # Check basic structure
        if not session.fiducial_pairs:
            logging.warning("Session has no fiducial pairs")
            return False
            
        # Check transformation parameters
        if not session.transformation.optimization_successful:
            logging.warning("Session transformation not optimized")
            return False
            
        if session.transformation.rms_error <= 0:
            logging.warning("Invalid RMS error in session")
            return False
            
        # Check coordinate dimensions
        for fid in session.fiducial_pairs:
            if len(fid.initial_3d) != 3 or len(fid.final_2d) != 2:
                logging.warning("Invalid coordinate dimensions in fiducial pair")
                return False
                
        return True
    except Exception as e:
        logging.error(f"Error validating session: {e}")
        return False

def train_e3nn_correlator(
    data_dir: str,
    synthetic_ratio: float = 0.7,
    n_synthetic_sessions: int = 10,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "logs"
):
    """Train the E3NN correlator on mixed real and synthetic data"""
    
    print("\n=== Starting E3NN Correlator Training ===")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Synthetic ratio: {synthetic_ratio}")
    print(f"Number of synthetic sessions: {n_synthetic_sessions}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Setup logging
    log_file = setup_logging(log_dir)
    print(f"Log file created at: {log_file}")
    
    logging.info("="*50)
    logging.info("Starting E3NN Correlator Training")
    logging.info("="*50)
    logging.info(f"Training parameters:")
    logging.info(f"  Data directory: {data_dir}")
    logging.info(f"  Synthetic ratio: {synthetic_ratio}")
    logging.info(f"  Number of synthetic sessions: {n_synthetic_sessions}")
    logging.info(f"  Number of epochs: {num_epochs}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Log file: {log_file}")
    logging.info("="*50)
    
    # Load real data with error handling
    print("\n=== Loading Real Data ===")
    try:
        parser = ThreeDCTDataParser()
        print("Parser initialized")
        
        real_sessions = parser.load_multiple_sessions(data_dir)
        print(f"Found {len(real_sessions)} real sessions")
        
        if not real_sessions:
            print(f"Warning: No valid sessions found in {data_dir}")
            real_sessions = []
        else:
            print(f"Successfully loaded {len(real_sessions)} real sessions")
            
            # Validate real sessions
            valid_real_sessions = []
            for i, session in enumerate(real_sessions):
                print(f"\nValidating real session {i+1}/{len(real_sessions)}")
                if validate_session_format(session):
                    valid_real_sessions.append(session)
                    print(f"✓ Session {session.session_id} validated")
                else:
                    print(f"✗ Session {session.session_id} failed validation")
            
            real_sessions = valid_real_sessions
            print(f"\nValidated {len(real_sessions)} real sessions")
            
    except Exception as e:
        print(f"\nError loading real data: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        real_sessions = []
    
    # Generate synthetic data with error handling
    print("\n=== Generating Synthetic Data ===")
    try:
        print("Initializing synthetic data generator...")
        generator = ThreeDCTSyntheticGenerator.from_real_data(data_dir)
        print("Generator initialized")
        
        synthetic_sessions = []
        print(f"\nGenerating {n_synthetic_sessions} synthetic sessions...")
        
        for i in range(n_synthetic_sessions):
            try:
                print(f"\nGenerating synthetic session {i+1}/{n_synthetic_sessions}")
                session = generator.generate_correlation_session(seed=100+i)
                if validate_session_format(session):
                    synthetic_sessions.append(session)
                    print(f"✓ Session {i+1} generated and validated")
                else:
                    print(f"✗ Session {i+1} failed validation")
            except Exception as e:
                print(f"Error generating synthetic session {i+1}: {str(e)}")
                continue
        
        print(f"\nSuccessfully generated {len(synthetic_sessions)} synthetic sessions")
        
    except Exception as e:
        print(f"\nError in synthetic data generation: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        synthetic_sessions = []
    
    # Check if we have enough data
    if not real_sessions and not synthetic_sessions:
        print("\nError: No valid training data available")
        raise ValueError("No valid training data available")
    
    # Create mixed dataset
    print("\n=== Creating Dataset ===")
    all_sessions = real_sessions + synthetic_sessions
    print(f"Total sessions: {len(all_sessions)}")
    print(f"  - Real sessions: {len(real_sessions)}")
    print(f"  - Synthetic sessions: {len(synthetic_sessions)}")
    
    try:
        print("\nInitializing dataset...")
        dataset = ThreeDCTDataset(all_sessions, parser=parser, normalize=True)
        print(f"Dataset created with {len(dataset)} training pairs")
        
        # Create dataloader
        print("\nCreating dataloader...")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        print(f"Dataloader created with batch size 32")
        
        # Initialize model
        print("\nInitializing model...")
        model = E3NNFiducialCorrelator(
            max_degree=2,
            hidden_features=64,
            num_layers=3,
            max_radius=5.0,
            num_neighbors=8
        ).to(device)
        print("Model initialized and moved to device")
        
        # Log model architecture
        print("\nModel architecture:")
        print(f"  Max degree: 2")
        print(f"  Hidden features: 64")
        print(f"  Number of layers: 3")
        print(f"  Max radius: 5.0")
        print(f"  Number of neighbors: 8")
        
        # Loss and optimizer
        print("\nSetting up loss function and optimizer...")
        criterion = GeometricLoss(
            mse_weight=1.0,
            distance_weight=0.2,
            angle_weight=0.1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Loss function and optimizer initialized")
        
        # Training loop
        print("\n=== Starting Training Loop ===")
        model.train()
        loss_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss = 0.0
            total_mse = 0.0
            batch_count = 0
            successful_batches = 0
            
            for batch_idx, (positions_3d, target_2d) in enumerate(dataloader):
                try:
                    print(f"\rProcessing batch {batch_idx+1}/{len(dataloader)}", end="")
                    
                    positions_3d = positions_3d.to(device)
                    target_2d = target_2d.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    pred_2d = model(positions_3d)
                    
                    # Compute loss
                    loss_dict = criterion(pred_2d, target_2d, positions_3d)
                    loss = loss_dict['total']
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_mse += loss_dict['mse'].item()
                    batch_count += 1
                    successful_batches += 1
                    
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            avg_mse = total_mse / batch_count if batch_count > 0 else float('inf')
            loss_history.append(avg_loss)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average MSE: {avg_mse:.4f}")
            print(f"  Successful batches: {successful_batches}/{len(dataloader)}")
        
        print("\n=== Training Completed ===")
        print(f"Final loss: {loss_history[-1]:.4f}")
        
        return model, loss_history
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise


def evaluate_model(model: E3NNFiducialCorrelator,
                   dataset: ThreeDCTDataset,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluate the trained model"""
    print("\n=== Evaluating Model ===")
    model.eval()
    total_error = 0.0
    total_samples = 0
    
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    
    with torch.no_grad():
        for i, (positions_3d, target_2d) in enumerate(dataset):
            try:
                print(f"\rProcessing sample {i+1}/{len(dataset)}", end="")
                
                positions_3d = positions_3d.to(device)
                target_2d = target_2d.to(device)
                
                pred_2d = model(positions_3d)
                
                # Calculate error in pixels
                error = torch.mean(torch.norm(pred_2d - target_2d, dim=-1))
                total_error += error.item()
                total_samples += 1
                
            except Exception as e:
                print(f"\nError processing sample {i}: {str(e)}")
                continue
    
    avg_error = total_error / total_samples
    print(f"\n\nEvaluation Results:")
    print(f"Total samples processed: {total_samples}")
    print(f"Average pixel error: {avg_error:.2f}")
    
    return avg_error


def visualize_predictions(model: E3NNFiducialCorrelator,
                         dataset: ThreeDCTDataset,
                         session_idx: int = 0,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Visualize model predictions"""
    print("\n=== Visualizing Predictions ===")
    print(f"Device: {device}")
    print(f"Session index: {session_idx}")
    
    model.eval()
    
    try:
        print("Loading data...")
        with torch.no_grad():
            positions_3d, target_2d = dataset[session_idx]
            positions_3d = positions_3d.to(device)
            
            print("Running model prediction...")
            pred_2d = model(positions_3d)
            
            # Convert back to numpy
            print("Converting to numpy arrays...")
            positions_3d = positions_3d.cpu().numpy().squeeze()
            target_2d = target_2d.cpu().numpy().squeeze()
            pred_2d = pred_2d.cpu().numpy().squeeze()
            
            # Denormalize if needed
            if dataset.normalize:
                print("Denormalizing data...")
                positions_3d = positions_3d * dataset.std_3d.numpy() + dataset.mean_3d.numpy()
                target_2d = target_2d * dataset.std_2d.numpy() + dataset.mean_2d.numpy()
                pred_2d = pred_2d * dataset.std_2d.numpy() + dataset.mean_2d.numpy()
        
        print("Creating visualization...")
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 3D points
        ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], 
                   c=positions_3d[:, 2], cmap='viridis', s=100)
        ax1.set_title('3D Fluorescence Points (colored by Z)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Target 2D
        ax2.scatter(target_2d[:, 0], target_2d[:, 1], c='blue', s=100, label='Target')
        ax2.set_title('Target 2D SEM Coordinates')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        # Prediction vs target
        ax3.scatter(target_2d[:, 0], target_2d[:, 1], c='blue', s=100, label='Target')
        ax3.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', s=100, label='Predicted')
        
        # Draw arrows showing errors
        for i in range(len(target_2d)):
            ax3.arrow(target_2d[i, 0], target_2d[i, 1],
                     pred_2d[i, 0] - target_2d[i, 0],
                     pred_2d[i, 1] - target_2d[i, 1],
                     head_width=10, head_length=10, fc='gray', ec='gray', alpha=0.7)
        
        ax3.set_title('Prediction vs Target')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Calculate errors
        print("\nCalculating errors...")
        errors = np.linalg.norm(pred_2d - target_2d, axis=1)
        print(f"Individual errors: {errors}")
        print(f"Mean error: {np.mean(errors):.2f} pixels")
        print(f"RMS error: {np.sqrt(np.mean(errors**2)):.2f} pixels")
        
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()


# Main execution example
if __name__ == "__main__":
    # Example usage
    data_dir = "data/3D_correlation_test_dataset"
    
    # Train the model
    print("Training E3NN correlator...")
    model, loss_history = train_e3nn_correlator(
        data_dir=data_dir,
        num_epochs=50,
        learning_rate=1e-3
    )
    
    # Save the model
    torch.save(model.state_dict(), "e3nn_correlator.pth")
    
    # Evaluate
    parser = ThreeDCTDataParser()
    sessions = parser.load_multiple_sessions(data_dir)
    dataset = ThreeDCTDataset(sessions, normalize=True)
    
    print("\nEvaluating model...")
    avg_error = evaluate_model(model, dataset)
    
    # Visualize
    print("\nVisualizing results...")
    visualize_predictions(model, dataset, session_idx=0)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()