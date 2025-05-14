# src/physics/simulator.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class SimulationParams:
    """Configuration for fiducial simulation"""
    # Real data parameters
    pixel_size_xy: float = 0.11  # μm (110 nm)
    pixel_size_z: float = 0.342  # μm (342 nm)
    grid_size: Tuple[float, float, float] = (20.0, 20.0, 5.0)  # μm
    ice_thickness: float = 1.0  # μm
    
    # Fiducial parameters
    bead_diameter: float = 1.0  # μm
    n_fiducials: int = 20  # Typical number in real data
    
    # Physics parameters
    gravity_strength: float = 0.8  # Bias toward bottom surface (0-1)
    surface_attraction: float = 0.3  # Attraction to grid holes
    
    # Grid parameters
    hole_spacing: float = 2.0  # μm
    hole_radius: float = 0.5  # μm
    
    # Transformation parameters (based on real data)
    typical_rotation: Tuple[float, float, float] = (90.0, 0.0, 25.0)  # degrees
    typical_scale: float = 1.0
    typical_rms_error: float = 1.0  # pixels

class FiducialSimulator:
    """Simulates realistic placement of fiducial beads on cryo-EM grids"""
    
    def __init__(self, params: SimulationParams = None):
        self.params = params or SimulationParams()
        
    def generate_correlation_data(
        self,
        seed: Optional[int] = None
    ) -> Dict:
        """Generate complete correlation data matching real format"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate initial 3D positions
        initial_3d = self.generate_fiducials(
            n_fiducials=self.params.n_fiducials,
            grid_pattern="holey_carbon",
            seed=seed
        )
        
        # Generate transformation parameters
        transform = self._generate_transformation()
        
        # Apply transformation to get 2D positions
        transformed_3d, final_2d, errors = self._apply_transformation(
            initial_3d, transform
        )
        
        # Generate POIs (Points of Interest)
        pois = self._generate_pois()
        
        return {
            'initial_3d': initial_3d,
            'transformed_3d': transformed_3d,
            'final_2d': final_2d,
            'errors': errors,
            'transformation': transform,
            'pois': pois,
            'rms_error': np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        }
    
    def _generate_transformation(self) -> Dict:
        """Generate realistic transformation parameters"""
        # Add small random variations to typical values
        rotation = np.array(self.params.typical_rotation) + np.random.normal(0, 2, 3)
        scale = self.params.typical_scale * (1 + np.random.normal(0, 0.02))
        
        # Generate translations
        trans_origin = np.random.normal(2000, 200, 3)  # Based on real data
        trans_center = np.random.normal(600, 100, 3)   # Based on real data
        
        return {
            'rotation_euler': rotation,
            'scale': scale,
            'translation_origin': trans_origin,
            'translation_center': trans_center,
            'center_point': np.array([828.0, 828.0, 828.0])  # From real data
        }
    
    def _apply_transformation(
        self,
        initial_3d: torch.Tensor,
        transform: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply transformation to get 2D positions and errors"""
        # Convert to numpy for easier matrix operations
        points = initial_3d.numpy()
        
        # Apply rotation
        phi, psi, theta = np.radians(transform['rotation_euler'])
        R = self._euler_to_rotation_matrix(phi, psi, theta)
        
        # Apply scale and rotation
        transformed = transform['scale'] * (R @ points.T).T
        
        # Apply translations
        transformed += transform['translation_center']
        
        # Project to 2D (drop Z coordinate)
        final_2d = transformed[:, :2]
        
        # Add realistic errors
        errors = np.random.normal(0, self.params.typical_rms_error/2, (len(points), 2))
        final_2d += errors
        
        return torch.from_numpy(transformed), torch.from_numpy(final_2d), torch.from_numpy(errors)
    
    def _euler_to_rotation_matrix(self, phi: float, psi: float, theta: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
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
        
        return R_z @ R_y @ R_x
    
    def _generate_pois(self) -> torch.Tensor:
        """Generate Points of Interest (non-fiducial features)"""
        n_pois = np.random.randint(2, 5)  # Typical number of POIs
        pois = []
        
        for _ in range(n_pois):
            # Generate random position within grid bounds
            x = np.random.uniform(0, self.params.grid_size[0])
            y = np.random.uniform(0, self.params.grid_size[1])
            z = np.random.uniform(0, self.params.grid_size[2])
            pois.append([x, y, z])
        
        return torch.tensor(pois)
    
    def generate_fiducials(
        self, 
        n_fiducials: int,
        grid_pattern: str = "holey_carbon",
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate realistic fiducial bead positions"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        positions = []
        grid_features = self._get_grid_features(grid_pattern)
        
        for i in range(n_fiducials):
            for attempt in range(self.params.max_attempts):
                # Start with random position
                pos = self._initial_random_position()
                
                # Apply physics constraints
                pos = self._apply_gravity(pos)
                pos = self._apply_surface_attraction(pos, grid_features)
                pos = self._ensure_bounds(pos)
                
                # Check collisions
                if self._check_collisions(pos, positions):
                    positions.append(pos)
                    break
            else:
                print(f"Warning: Could only place {len(positions)} of {n_fiducials} fiducials")
                break
        
        return torch.stack(positions) if positions else torch.empty(0, 3)
    
    def _initial_random_position(self) -> torch.Tensor:
        """Generate initial random position within grid bounds"""
        x = torch.rand(1) * self.params.grid_size[0]
        y = torch.rand(1) * self.params.grid_size[1]
        z = torch.rand(1) * self.params.grid_size[2]
        return torch.tensor([x.item(), y.item(), z.item()])
    
    def _check_collisions(
        self, 
        position: torch.Tensor, 
        existing: List[torch.Tensor]
    ) -> bool:
        """Check if new position collides with existing beads"""
        if not existing:
            return True
        
        existing_tensor = torch.stack(existing)
        distances = torch.norm(position - existing_tensor, dim=1)
        return torch.all(distances > self.params.bead_diameter)
    
    def _apply_gravity(self, position: torch.Tensor) -> torch.Tensor:
        """Apply gravity bias toward bottom surface"""
        pos = position.clone()
        # Bias z-coordinate toward 0 (bottom surface)
        gravity_bias = torch.rand(1) * self.params.gravity_strength
        pos[2] = pos[2] * (1 - gravity_bias)
        return pos
    
    def _apply_surface_attraction(
        self, 
        position: torch.Tensor,
        grid_features: List[Tuple[torch.Tensor, float]]
    ) -> torch.Tensor:
        """Apply attraction to grid features (holes, bars)"""
        pos = position.clone()
        
        # Find closest attractive feature
        min_distance = float('inf')
        closest_feature = None
        
        for feature_pos, attraction_strength in grid_features:
            distance = torch.norm(pos[:2] - feature_pos[:2])  # Only x,y distance
            if distance < min_distance:
                min_distance = distance
                closest_feature = (feature_pos, attraction_strength)
        
        if closest_feature and min_distance < self.params.hole_radius * 2:
            feature_pos, strength = closest_feature
            # Pull toward the feature center
            direction = feature_pos[:2] - pos[:2]
            pull_strength = strength * self.params.surface_attraction
            pos[:2] = pos[:2] + direction * pull_strength
        
        return pos
    
    def _get_grid_features(self, grid_pattern: str) -> List[Tuple[torch.Tensor, float]]:
        """Define attractive features for different grid types"""
        features = []
        
        if grid_pattern == "holey_carbon":
            # Create a regular grid of holes
            hole_spacing = self.params.hole_spacing
            x_holes = int(self.params.grid_size[0] / hole_spacing) + 1
            y_holes = int(self.params.grid_size[1] / hole_spacing) + 1
            
            for i in range(x_holes):
                for j in range(y_holes):
                    x = i * hole_spacing
                    y = j * hole_spacing
                    # Only add if within bounds
                    if x <= self.params.grid_size[0] and y <= self.params.grid_size[1]:
                        feature_pos = torch.tensor([x, y, 0.0])
                        # Higher attraction for central holes
                        center_x = self.params.grid_size[0] / 2
                        center_y = self.params.grid_size[1] / 2
                        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        attraction = 1.0 - min(dist_from_center / (center_x + center_y), 0.5)
                        features.append((feature_pos, attraction))
        
        elif grid_pattern == "lacey_carbon":
            # Random holes with clustering
            n_holes = int(self.params.grid_size[0] * self.params.grid_size[1] / 8)
            for _ in range(n_holes):
                x = torch.rand(1) * self.params.grid_size[0]
                y = torch.rand(1) * self.params.grid_size[1]
                feature_pos = torch.tensor([x.item(), y.item(), 0.0])
                features.append((feature_pos, 0.7))
        
        return features
    
    def _ensure_bounds(self, position: torch.Tensor) -> torch.Tensor:
        """Ensure position stays within grid bounds"""
        pos = position.clone()
        pos[0] = torch.clamp(pos[0], self.params.bead_diameter/2, 
                           self.params.grid_size[0] - self.params.bead_diameter/2)
        pos[1] = torch.clamp(pos[1], self.params.bead_diameter/2, 
                           self.params.grid_size[1] - self.params.bead_diameter/2)
        pos[2] = torch.clamp(pos[2], self.params.bead_diameter/2, 
                           self.params.grid_size[2] - self.params.bead_diameter/2)
        return pos
    
    def validate_simulation(self, positions: torch.Tensor) -> Dict[str, bool]:
        """Validate that simulation results are realistic"""
        results = {}
        
        # Check no overlaps
        pairwise_distances = torch.cdist(positions, positions)
        # Set diagonal to large value to ignore self-distances
        pairwise_distances.fill_diagonal_(float('inf'))
        min_distance = pairwise_distances.min()
        results['no_overlaps'] = min_distance >= self.params.bead_diameter
        
        # Check reasonable z-distribution (more near bottom)
        z_values = positions[:, 2]
        bottom_half = (z_values < self.params.grid_size[2] / 2).sum()
        total = len(z_values)
        results['gravity_bias'] = (bottom_half / total) > 0.6 if total > 0 else True
        
        # Check clustering around features (for validation)
        if len(positions) > 1:
            # Calculate average nearest neighbor distance
            nn_distances = []
            for i, pos in enumerate(positions):
                others = torch.cat([positions[:i], positions[i+1:]])
                if len(others) > 0:
                    distances = torch.norm(pos - others, dim=1)
                    nn_distances.append(distances.min().item())
            
            avg_nn_distance = np.mean(nn_distances)
            # Realistic if beads aren't too spread out
            results['realistic_clustering'] = avg_nn_distance < self.params.grid_size[0] * 0.5
        else:
            results['realistic_clustering'] = True
        
        return results
    
    def visualize(self, positions: torch.Tensor, grid_pattern: str = "holey_carbon"):
        """Visualize the generated fiducial positions"""
        fig = plt.figure(figsize=(12, 5))
        
        # 3D view
        ax1 = fig.add_subplot(121, projection='3d')
        if len(positions) > 0:
            ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                       c='red', s=100, alpha=0.8)
        
        # Add grid features for visualization
        features = self._get_grid_features(grid_pattern)
        for feature_pos, _ in features:
            ax1.scatter(feature_pos[0], feature_pos[1], feature_pos[2], 
                       c='blue', s=50, alpha=0.3, marker='s')
        
        ax1.set_xlim(0, self.params.grid_size[0])
        ax1.set_ylim(0, self.params.grid_size[1])
        ax1.set_zlim(0, self.params.grid_size[2])
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        ax1.set_zlabel('Z (μm)')
        ax1.set_title('3D Fiducial Distribution')
        
        # Top-down view
        ax2 = fig.add_subplot(122)
        if len(positions) > 0:
            # Color by height
            scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                                c=positions[:, 2], cmap='viridis', s=100, alpha=0.8)
            plt.colorbar(scatter, ax=ax2, label='Z height (μm)')
        
        # Add grid features
        for feature_pos, _ in features:
            circle = plt.Circle((feature_pos[0], feature_pos[1]), 
                              self.params.hole_radius, fill=False, color='blue', alpha=0.3)
            ax2.add_patch(circle)
        
        ax2.set_xlim(0, self.params.grid_size[0])
        ax2.set_ylim(0, self.params.grid_size[1])
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        ax2.set_title('Top-down View (colored by height)')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create simulator with parameters matching real data
    params = SimulationParams(
        pixel_size_xy=0.11,
        pixel_size_z=0.342,
        grid_size=(20.0, 20.0, 5.0),
        n_fiducials=20,
        typical_rotation=(92.488, -1.326, 26.773),
        typical_scale=1.018
    )
    
    simulator = FiducialSimulator(params)
    
    # Generate correlation data
    correlation_data = simulator.generate_correlation_data(seed=42)
    
    print("=== Generated Correlation Data ===")
    print(f"Number of fiducials: {len(correlation_data['initial_3d'])}")
    print(f"Number of POIs: {len(correlation_data['pois'])}")
    print(f"RMS error: {correlation_data['rms_error']:.2f} pixels")
    print("\nTransformation parameters:")
    for key, value in correlation_data['transformation'].items():
        print(f"{key}: {value}")
    
    # Visualize
    simulator.visualize(correlation_data['initial_3d'], "holey_carbon")
