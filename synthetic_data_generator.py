# src/synthetic_data/tdct_synthetic_generator.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_analysis.tdct_data_parser import (
    TransformationParams, FiducialPair, POI, CorrelationSession,
    ThreeDCTDataParser
)

@dataclass
class SyntheticParams:
    """Parameters for synthetic 3DCT data generation, empirically derived from real data"""
    
    # Coordinate system bounds (learned from real data)
    fluorescence_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    sem_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    
    # Transformation parameter ranges (learned from real data)
    rotation_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    scale_range: Tuple[float, float]
    translation_origin_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    translation_center_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    center_point: Tuple[float, float, float]
    
    # Quality parameters (learned from real data)
    rms_error_range: Tuple[float, float]
    error_std_per_fiducial: float
    
    # Count parameters (learned from real data)
    n_fiducials_range: Tuple[int, int]
    n_pois_range: Tuple[int, int]
    
    # Physics constraints
    min_fiducial_distance: float = 30.0  # Minimum distance between fiducials
    max_placement_attempts: int = 1000

class EmpiricalParameterExtractor:
    """Extract parameters from real 3DCT data for synthetic generation"""
    
    def __init__(self, parser: ThreeDCTDataParser):
        print("\nInitializing EmpiricalParameterExtractor")
        self.parser = parser
    
    def extract_parameters_from_sessions(self, sessions: List[CorrelationSession]) -> SyntheticParams:
        """Extract empirical parameters from real correlation sessions"""
        print(f"\n=== Extracting Parameters from {len(sessions)} Real Sessions ===")
        
        # Extract coordinate bounds
        print("\nExtracting coordinate bounds...")
        fluorescence_bounds = self._extract_fluorescence_bounds(sessions)
        print(f"Fluorescence bounds: {fluorescence_bounds}")
        sem_bounds = self._extract_sem_bounds(sessions)
        print(f"SEM bounds: {sem_bounds}")
        
        # Extract transformation parameters
        print("\nExtracting transformation parameters...")
        rotation_ranges = self._extract_rotation_ranges(sessions)
        print(f"Rotation ranges: {rotation_ranges}")
        scale_range = self._extract_scale_range(sessions)
        print(f"Scale range: {scale_range}")
        translation_origin_ranges = self._extract_translation_origin_ranges(sessions)
        print(f"Translation origin ranges: {translation_origin_ranges}")
        translation_center_ranges = self._extract_translation_center_ranges(sessions)
        print(f"Translation center ranges: {translation_center_ranges}")
        center_point = self._extract_center_point(sessions)
        print(f"Center point: {center_point}")
        
        # Extract quality parameters
        print("\nExtracting quality parameters...")
        rms_error_range = self._extract_rms_error_range(sessions)
        print(f"RMS error range: {rms_error_range}")
        error_std_per_fiducial = self._extract_error_std(sessions)
        print(f"Error std per fiducial: {error_std_per_fiducial:.3f}")
        
        # Extract count parameters
        print("\nExtracting count parameters...")
        n_fiducials_range = self._extract_fiducial_count_range(sessions)
        print(f"Fiducial count range: {n_fiducials_range}")
        n_pois_range = self._extract_poi_count_range(sessions)
        print(f"POI count range: {n_pois_range}")
        
        params = SyntheticParams(
            fluorescence_bounds=fluorescence_bounds,
            sem_bounds=sem_bounds,
            rotation_ranges=rotation_ranges,
            scale_range=scale_range,
            translation_origin_ranges=translation_origin_ranges,
            translation_center_ranges=translation_center_ranges,
            center_point=center_point,
            rms_error_range=rms_error_range,
            error_std_per_fiducial=error_std_per_fiducial,
            n_fiducials_range=n_fiducials_range,
            n_pois_range=n_pois_range
        )
        
        self._print_extracted_parameters(params)
        return params
    
    def _extract_fluorescence_bounds(self, sessions: List[CorrelationSession]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Extract fluorescence coordinate bounds from real data"""
        all_fluor_coords = []
        
        for session in sessions:
            for fid in session.fiducial_pairs:
                all_fluor_coords.append(fid.initial_3d)
        
        if not all_fluor_coords:
            # Fallback defaults
            return ((200, 1600), (300, 1200), (40, 80))
        
        coords_array = np.array(all_fluor_coords)
        
        # Add small margin to bounds
        margin = 0.05  # 5% margin
        x_range = coords_array[:, 0].min(), coords_array[:, 0].max()
        y_range = coords_array[:, 1].min(), coords_array[:, 1].max()
        z_range = coords_array[:, 2].min(), coords_array[:, 2].max()
        
        x_margin = (x_range[1] - x_range[0]) * margin
        y_margin = (y_range[1] - y_range[0]) * margin
        z_margin = (z_range[1] - z_range[0]) * margin
        
        return (
            (x_range[0] - x_margin, x_range[1] + x_margin),
            (y_range[0] - y_margin, y_range[1] + y_margin),
            (z_range[0] - z_margin, z_range[1] + z_margin)
        )
    
    def _extract_sem_bounds(self, sessions: List[CorrelationSession]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Extract SEM coordinate bounds from real data"""
        all_sem_coords = []
        
        for session in sessions:
            for fid in session.fiducial_pairs:
                all_sem_coords.append(fid.final_2d)
        
        if not all_sem_coords:
            # Fallback defaults
            return ((400, 2100), (800, 1400))
        
        coords_array = np.array(all_sem_coords)
        
        # Add small margin to bounds
        margin = 0.05  # 5% margin
        x_range = coords_array[:, 0].min(), coords_array[:, 0].max()
        y_range = coords_array[:, 1].min(), coords_array[:, 1].max()
        
        x_margin = (x_range[1] - x_range[0]) * margin
        y_margin = (y_range[1] - y_range[0]) * margin
        
        return (
            (x_range[0] - x_margin, x_range[1] + x_margin),
            (y_range[0] - y_margin, y_range[1] + y_margin)
        )
    
    def _extract_rotation_ranges(self, sessions: List[CorrelationSession]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Extract rotation parameter ranges from real data"""
        rotations = [session.transformation.rotation_euler for session in sessions]
        
        if not rotations:
            # Fallback defaults
            return ((80, 100), (-10, 10), (15, 35))
        
        rotations_array = np.array(rotations)
        
        # Extract ranges for phi, psi, theta with some margin
        margin = 5.0  # 5 degrees margin
        phi_range = (rotations_array[:, 0].min() - margin, rotations_array[:, 0].max() + margin)
        psi_range = (rotations_array[:, 1].min() - margin, rotations_array[:, 1].max() + margin)
        theta_range = (rotations_array[:, 2].min() - margin, rotations_array[:, 2].max() + margin)
        
        return (phi_range, psi_range, theta_range)
    
    def _extract_scale_range(self, sessions: List[CorrelationSession]) -> Tuple[float, float]:
        """Extract scale parameter range from real data"""
        scales = [session.transformation.scale for session in sessions]
        
        if not scales:
            return (0.98, 1.02)
        
        margin = 0.01  # 1% margin
        return (min(scales) - margin, max(scales) + margin)
    
    def _extract_translation_origin_ranges(self, sessions: List[CorrelationSession]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Extract translation_origin parameter ranges from real data"""
        trans_origins = [session.transformation.translation_origin for session in sessions]
        
        if not trans_origins:
            return ((2000, 2500), (200, 350), (-50, 50))
        
        trans_array = np.array(trans_origins)
        
        # Extract ranges with margin
        margin = 0.1  # 10% margin
        x_range = trans_array[:, 0].min(), trans_array[:, 0].max()
        y_range = trans_array[:, 1].min(), trans_array[:, 1].max()
        z_range = trans_array[:, 2].min(), trans_array[:, 2].max()
        
        x_margin = max(abs(x_range[1] - x_range[0]) * margin, 100)
        y_margin = max(abs(y_range[1] - y_range[0]) * margin, 50)
        z_margin = max(abs(z_range[1] - z_range[0]) * margin, 25)
        
        return (
            (x_range[0] - x_margin, x_range[1] + x_margin),
            (y_range[0] - y_margin, y_range[1] + y_margin),
            (z_range[0] - z_margin, z_range[1] + z_margin)
        )
    
    def _extract_translation_center_ranges(self, sessions: List[CorrelationSession]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Extract translation_center parameter ranges from real data"""
        trans_centers = [session.transformation.translation_center for session in sessions]
        
        if not trans_centers:
            return ((500, 800), (100, 200), (200, 350))
        
        trans_array = np.array(trans_centers)
        
        # Extract ranges with margin
        margin = 0.1  # 10% margin
        x_range = trans_array[:, 0].min(), trans_array[:, 0].max()
        y_range = trans_array[:, 1].min(), trans_array[:, 1].max()
        z_range = trans_array[:, 2].min(), trans_array[:, 2].max()
        
        x_margin = max(abs(x_range[1] - x_range[0]) * margin, 50)
        y_margin = max(abs(y_range[1] - y_range[0]) * margin, 25)
        z_margin = max(abs(z_range[1] - z_range[0]) * margin, 25)
        
        return (
            (x_range[0] - x_margin, x_range[1] + x_margin),
            (y_range[0] - y_margin, y_range[1] + y_margin),
            (z_range[0] - z_margin, z_range[1] + z_margin)
        )
    
    def _extract_center_point(self, sessions: List[CorrelationSession]) -> Tuple[float, float, float]:
        """Extract center point from real data (should be consistent)"""
        center_points = [session.transformation.center_point for session in sessions]
        
        if not center_points:
            return (828.0, 828.0, 828.0)
        
        # Use mean center point (should be the same across sessions)
        center_array = np.array(center_points)
        mean_center = center_array.mean(axis=0)
        
        return tuple(mean_center)
    
    def _extract_rms_error_range(self, sessions: List[CorrelationSession]) -> Tuple[float, float]:
        """Extract RMS error range from real data"""
        rms_errors = [session.transformation.rms_error for session in sessions]
        
        if not rms_errors:
            return (0.5, 2.0)
        
        margin = 0.2  # Add 0.2 pixels margin
        return (max(0.1, min(rms_errors) - margin), max(rms_errors) + margin)
    
    def _extract_error_std(self, sessions: List[CorrelationSession]) -> float:
        """Extract standard deviation of individual fiducial errors"""
        all_errors = []
        
        for session in sessions:
            for fid in session.fiducial_pairs:
                error_magnitude = np.linalg.norm(fid.error)
                all_errors.append(error_magnitude)
        
        if not all_errors:
            return 0.8
        
        return np.std(all_errors)
    
    def _extract_fiducial_count_range(self, sessions: List[CorrelationSession]) -> Tuple[int, int]:
        """Extract fiducial count range from real data"""
        counts = [len(session.fiducial_pairs) for session in sessions]
        
        if not counts:
            return (8, 22)
        
        return (min(counts), max(counts))
    
    def _extract_poi_count_range(self, sessions: List[CorrelationSession]) -> Tuple[int, int]:
        """Extract POI count range from real data"""
        counts = [len(session.pois) for session in sessions]
        
        if not counts:
            return (2, 5)
        
        return (max(1, min(counts)), max(counts))
    
    def _print_extracted_parameters(self, params: SyntheticParams):
        """Print extracted parameters for verification"""
        print("\n=== Extracted Empirical Parameters ===")
        print(f"Fluorescence bounds: {params.fluorescence_bounds}")
        print(f"SEM bounds: {params.sem_bounds}")
        print(f"Rotation ranges: {params.rotation_ranges}")
        print(f"Scale range: {params.scale_range}")
        print(f"Translation origin ranges: {params.translation_origin_ranges}")
        print(f"Translation center ranges: {params.translation_center_ranges}")
        print(f"Center point: {params.center_point}")
        print(f"RMS error range: {params.rms_error_range}")
        print(f"Error std per fiducial: {params.error_std_per_fiducial:.3f}")
        print(f"Fiducial count range: {params.n_fiducials_range}")
        print(f"POI count range: {params.n_pois_range}")

class ThreeDCTSyntheticGenerator:
    """Generate synthetic 3DCT correlation data using empirically derived parameters"""
    
    def __init__(self, params: SyntheticParams):
        print("\nInitializing ThreeDCTSyntheticGenerator")
        self.params = params
        self.rng = np.random.RandomState()
        print("✓ Generator initialized with parameters")
    
    @classmethod
    def from_real_data(cls, data_dir: str):
        """Create generator by analyzing real 3DCT data"""
        print(f"\n=== Creating Generator from Real Data ===")
        print(f"Data directory: {data_dir}")
        
        parser = ThreeDCTDataParser()
        sessions = parser.load_multiple_sessions(data_dir)
        
        if not sessions:
            print(f"✗ No correlation sessions found in {data_dir}")
            raise ValueError(f"No correlation sessions found in {data_dir}")
        
        print(f"✓ Found {len(sessions)} real sessions")
        
        extractor = EmpiricalParameterExtractor(parser)
        params = extractor.extract_parameters_from_sessions(sessions)
        
        return cls(params)
    
    def generate_training_samples(
        self, 
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate training samples in the format expected by E3NN network"""
        print(f"\n=== Generating {n_samples} Training Samples ===")
        if seed is not None:
            print(f"Using seed: {seed}")
            self.rng.seed(seed)
        
        training_samples = []
        
        for i in range(n_samples):
            print(f"\nGenerating sample {i+1}/{n_samples}")
            session = self.generate_correlation_session(seed=seed+i if seed else None)
            training_pairs = ThreeDCTDataParser().extract_training_pairs(session)
            training_samples.extend(training_pairs)
            print(f"✓ Generated {len(training_pairs)} training pairs")
        
        print(f"\n=== Training Sample Generation Complete ===")
        print(f"Total training pairs: {len(training_samples)}")
        return training_samples
    
    def generate_correlation_session(
        self,
        seed: Optional[int] = None
    ) -> CorrelationSession:
        """Generate a complete synthetic correlation session matching real data format"""
        print(f"\n=== Generating Synthetic Correlation Session ===")
        if seed is not None:
            print(f"Using seed: {seed}")
            self.rng.seed(seed)
        
        # Generate number of fiducials
        n_fiducials = self.rng.randint(*self.params.n_fiducials_range)
        print(f"Generating {n_fiducials} fiducials")
        
        # Generate 3D fluorescence positions
        print("\nGenerating 3D fluorescence positions...")
        fluorescence_3d = self._generate_realistic_3d_positions(n_fiducials)
        print(f"✓ Generated {len(fluorescence_3d)} 3D positions")
        
        # Generate transformation parameters
        print("\nGenerating transformation parameters...")
        transform_params = self._generate_transformation_parameters()
        print(f"Rotation: {transform_params.rotation_euler}")
        print(f"Scale: {transform_params.scale:.3f}")
        
        # Apply transformation
        print("\nApplying 3D to 2D transformation...")
        sem_2d, individual_errors = self._apply_3d_to_2d_transformation(
            fluorescence_3d, transform_params
        )
        print(f"✓ Transformed to 2D coordinates")
        
        # Create fiducial pairs
        print("\nCreating fiducial pairs...")
        fiducial_pairs = []
        for i in range(n_fiducials):
            transformed_3d = self._transform_3d_point(fluorescence_3d[i], transform_params)
            fiducial_pairs.append(FiducialPair(
                initial_3d=fluorescence_3d[i],
                final_2d=sem_2d[i],
                transformed_3d=transformed_3d,
                error=individual_errors[i],
                index=i
            ))
        print(f"✓ Created {len(fiducial_pairs)} fiducial pairs")
        
        # Generate POIs
        print("\nGenerating points of interest...")
        pois = self._generate_pois()
        print(f"✓ Generated {len(pois)} POIs")
        
        # Calculate RMS error
        rms_error = np.sqrt(np.mean(np.sum(individual_errors**2, axis=1)))
        transform_params.rms_error = rms_error
        print(f"Calculated RMS error: {rms_error:.2f} pixels")
        
        # Create session
        session_id = f"synthetic_{seed}" if seed else f"synthetic_{self.rng.randint(10000)}"
        print(f"\nCreating session with ID: {session_id}")
        
        return CorrelationSession(
            session_id=session_id,
            machine_info="VM-SYNTHETIC ['Python', 'x64', 'Synthetic Data Generator']",
            date="Wed Jan 01 12:00:00 2024",
            transformation=transform_params,
            fiducial_pairs=fiducial_pairs,
            pois=pois,
            csv_coordinates=None,
            microscope_center=np.array([1536.0, 1024.0]),
            pixel_size_um=None
        )
    
    def _generate_realistic_3d_positions(self, n_fiducials: int) -> np.ndarray:
        """Generate realistic 3D fluorescence positions using empirical bounds"""
        positions = []
        
        # Get empirical bounds
        x_bounds, y_bounds, z_bounds = self.params.fluorescence_bounds
        
        for _ in range(n_fiducials):
            for attempt in range(self.params.max_placement_attempts):
                # Generate candidate position within empirical bounds
                x = self.rng.uniform(*x_bounds)
                y = self.rng.uniform(*y_bounds)
                z = self.rng.uniform(*z_bounds)
                
                candidate = np.array([x, y, z])
                
                # Check minimum distance constraint
                if self._check_minimum_distance(candidate, positions):
                    positions.append(candidate)
                    break
            else:
                # If we can't place all fiducials, use what we have
                print(f"Warning: Could only place {len(positions)} of {n_fiducials} fiducials")
                break
        
        return np.array(positions)
    
    def _check_minimum_distance(
        self, 
        candidate: np.ndarray, 
        existing: List[np.ndarray]
    ) -> bool:
        """Check if candidate position satisfies minimum distance constraint"""
        if not existing:
            return True
        
        existing_array = np.array(existing)
        distances = np.linalg.norm(existing_array - candidate, axis=1)
        return np.all(distances >= self.params.min_fiducial_distance)
    
    def _generate_transformation_parameters(self) -> TransformationParams:
        """Generate realistic transformation parameters using empirical ranges"""
        
        # Generate rotation angles using empirical ranges
        phi = self.rng.uniform(*self.params.rotation_ranges[0])
        psi = self.rng.uniform(*self.params.rotation_ranges[1])
        theta = self.rng.uniform(*self.params.rotation_ranges[2])
        rotation_euler = np.array([phi, psi, theta])
        
        # Generate scale using empirical range
        scale = self.rng.uniform(*self.params.scale_range)
        
        # Generate translations using empirical ranges
        trans_origin_x = self.rng.uniform(*self.params.translation_origin_ranges[0])
        trans_origin_y = self.rng.uniform(*self.params.translation_origin_ranges[1])
        trans_origin_z = self.rng.uniform(*self.params.translation_origin_ranges[2])
        translation_origin = np.array([trans_origin_x, trans_origin_y, trans_origin_z])
        
        trans_center_x = self.rng.uniform(*self.params.translation_center_ranges[0])
        trans_center_y = self.rng.uniform(*self.params.translation_center_ranges[1])
        trans_center_z = self.rng.uniform(*self.params.translation_center_ranges[2])
        translation_center = np.array([trans_center_x, trans_center_y, trans_center_z])
        
        # Use empirical center point
        center_point = np.array(self.params.center_point)
        
        return TransformationParams(
            rotation_euler=rotation_euler,
            scale=scale,
            translation_origin=translation_origin,
            translation_center=translation_center,
            center_point=center_point,
            rms_error=0.0,  # Will be calculated later
            optimization_successful=True
        )
    
    def _apply_3d_to_2d_transformation(
        self,
        points_3d: np.ndarray,
        transform_params: TransformationParams
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 3D→2D transformation exactly matching 3DCT's method"""
        
        # Convert points_3d relative to center point for rotation
        center = transform_params.center_point
        points_centered = points_3d - center
        
        # Convert Euler angles to rotation matrix
        phi, psi, theta = np.radians(transform_params.rotation_euler)
        R = self._euler_to_rotation_matrix(phi, psi, theta)
        
        # Apply transformation: scale and rotate the centered points
        transformed_centered = transform_params.scale * (R @ points_centered.T).T
        
        # Move back to original coordinate system and apply translation
        transformed_3d = transformed_centered + center + transform_params.translation_center
        
        # Project to 2D (extract x, y coordinates)
        # This matches the "Transformed initial" columns in the TXT file
        projected_2d = transformed_3d[:, :2]
        
        # Ensure projected points are within SEM bounds
        projected_2d = self._constrain_to_sem_bounds(projected_2d)
        
        # Add realistic per-fiducial errors using empirical error distribution
        individual_errors = self._generate_realistic_errors(len(points_3d))
        final_2d = projected_2d + individual_errors
        
        return final_2d, individual_errors
    
    def _constrain_to_sem_bounds(self, points_2d: np.ndarray) -> np.ndarray:
        """Adjust transformation to ensure points fall within empirical SEM bounds"""
        x_bounds, y_bounds = self.params.sem_bounds
        
        # Calculate current bounds
        current_x_min, current_x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        current_y_min, current_y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
        
        # Calculate required shifts to fit within bounds
        shift_x = 0
        shift_y = 0
        
        if current_x_min < x_bounds[0]:
            shift_x = x_bounds[0] - current_x_min + 20  # Add margin
        elif current_x_max > x_bounds[1]:
            shift_x = x_bounds[1] - current_x_max - 20  # Add margin
        
        if current_y_min < y_bounds[0]:
            shift_y = y_bounds[0] - current_y_min + 20  # Add margin
        elif current_y_max > y_bounds[1]:
            shift_y = y_bounds[1] - current_y_max - 20  # Add margin
        
        # Apply shifts
        adjusted_points = points_2d.copy()
        adjusted_points[:, 0] += shift_x
        adjusted_points[:, 1] += shift_y
        
        return adjusted_points
    
    def _generate_realistic_errors(self, n_fiducials: int) -> np.ndarray:
        """Generate realistic per-fiducial errors using empirical distribution"""
        # Generate errors using empirical error standard deviation
        errors = self.rng.normal(
            0, 
            self.params.error_std_per_fiducial, 
            (n_fiducials, 2)
        )
        
        # Scale errors to match empirical RMS range
        target_rms = self.rng.uniform(*self.params.rms_error_range)
        current_rms = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        # Scale errors to match target RMS
        if current_rms > 0:
            errors = errors * (target_rms / current_rms)
        
        return errors
    
    def _transform_3d_point(
        self,
        point_3d: np.ndarray,
        transform_params: TransformationParams
    ) -> np.ndarray:
        """Transform a single 3D point exactly matching 3DCT method"""
        # Center the point for rotation
        center = transform_params.center_point
        point_centered = point_3d - center
        
        # Apply rotation and scale
        phi, psi, theta = np.radians(transform_params.rotation_euler)
        R = self._euler_to_rotation_matrix(phi, psi, theta)
        
        transformed_centered = transform_params.scale * (R @ point_centered)
        
        # Move back and apply translation
        transformed = transformed_centered + center + transform_params.translation_center
        
        return transformed
    
    def _euler_to_rotation_matrix(self, phi: float, psi: float, theta: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix (matching 3DCT convention)"""
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
        
        # 3DCT uses: R = R_z * R_y * R_x
        return R_z @ R_y @ R_x
    
    def _generate_pois(self) -> List[POI]:
        """Generate Points of Interest using empirical distributions"""
        n_pois = self.rng.randint(*self.params.n_pois_range)
        pois = []
        
        for _ in range(n_pois):
            # Generate 3D position within empirical fluorescence bounds
            x_bounds, y_bounds, z_bounds = self.params.fluorescence_bounds
            x = self.rng.uniform(*x_bounds)
            y = self.rng.uniform(*y_bounds)
            z = self.rng.uniform(*z_bounds)
            initial_3d = np.array([x, y, z])
            
            # Generate corresponding 2D+Z position within empirical SEM bounds
            sem_x_bounds, sem_y_bounds = self.params.sem_bounds
            sem_x = self.rng.uniform(*sem_x_bounds)
            sem_y = self.rng.uniform(*sem_y_bounds)
            # Z coordinate for POIs typically ranges from 450-490 (empirical observation)
            sem_z = self.rng.uniform(460, 490)
            correlated_2d = np.array([sem_x, sem_y, sem_z])
            
            # Generate distances from SEM center (1536.0, 1024.0)
            # Calculate actual distances to the generated correlated spot
            sem_center = np.array([1536.0, 1024.0])
            distance_px = np.array([sem_x, sem_y]) - sem_center
            # Distance in um is often 'nan' in real data, but we can estimate
            # Assuming pixel size around 0.1 um/pixel (typical for SEM)
            distance_um = distance_px * 0.1
            
            pois.append(POI(
                initial_3d=initial_3d,
                correlated_2d=correlated_2d,
                distance_px=distance_px,
                distance_um=distance_um
            ))
        
        return pois
    
    def create_mixed_dataset(
        self,
        n_synthetic_sessions: int,
        real_sessions: List[CorrelationSession],
        synthetic_ratio: float = 0.7,
        seed: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create a mixed dataset of synthetic and real data"""
        print(f"\n=== Creating Mixed Dataset ===")
        print(f"Number of synthetic sessions: {n_synthetic_sessions}")
        print(f"Number of real sessions: {len(real_sessions)}")
        print(f"Synthetic ratio: {synthetic_ratio}")
        
        if seed is not None:
            print(f"Using seed: {seed}")
            self.rng.seed(seed)
        
        # Generate synthetic data
        print("\nGenerating synthetic data...")
        synthetic_samples = []
        for i in range(n_synthetic_sessions):
            print(f"\nGenerating synthetic session {i+1}/{n_synthetic_sessions}")
            session = self.generate_correlation_session(seed=seed+i if seed else None)
            pairs = ThreeDCTDataParser().extract_training_pairs(session)
            synthetic_samples.extend(pairs)
            print(f"✓ Generated {len(pairs)} pairs")
        
        # Extract real data
        print("\nExtracting real data...")
        real_samples = []
        parser = ThreeDCTDataParser()
        for i, session in enumerate(real_sessions):
            print(f"\nProcessing real session {i+1}/{len(real_sessions)}")
            pairs = parser.extract_training_pairs(session)
            real_samples.extend(pairs)
            print(f"✓ Extracted {len(pairs)} pairs")
        
        # Calculate sample sizes
        total_synthetic = len(synthetic_samples)
        total_real = len(real_samples)
        
        if total_real == 0:
            print("\nNo real data available, using only synthetic data")
            return synthetic_samples
        
        # Determine target numbers
        target_total = int(total_real / (1 - synthetic_ratio))
        target_synthetic = int(target_total * synthetic_ratio)
        target_real = target_total - target_synthetic
        
        print(f"\nTarget sample sizes:")
        print(f"  Total: {target_total}")
        print(f"  Synthetic: {target_synthetic}")
        print(f"  Real: {target_real}")
        
        # Sample the data
        print("\nSampling data...")
        selected_synthetic = synthetic_samples
        if len(synthetic_samples) > target_synthetic:
            indices = self.rng.choice(len(synthetic_samples), target_synthetic, replace=False)
            selected_synthetic = [synthetic_samples[i] for i in indices]
            print(f"Selected {len(selected_synthetic)} synthetic samples")
        
        selected_real = real_samples
        if len(real_samples) > target_real:
            indices = self.rng.choice(len(real_samples), target_real, replace=False)
            selected_real = [real_samples[i] for i in indices]
            print(f"Selected {len(selected_real)} real samples")
        
        # Combine and shuffle
        print("\nCombining and shuffling dataset...")
        all_samples = selected_synthetic + selected_real
        self.rng.shuffle(all_samples)
        
        print(f"\n=== Mixed Dataset Created ===")
        print(f"  Synthetic samples: {len(selected_synthetic)}")
        print(f"  Real samples: {len(selected_real)}")
        print(f"  Total samples: {len(all_samples)}")
        
        return all_samples
    
    def visualize_synthetic_session(self, session: CorrelationSession):
        """Visualize a synthetic correlation session"""
        fig = plt.figure(figsize=(15, 5))
        
        # Extract data
        fluorescence_3d = np.array([fp.initial_3d for fp in session.fiducial_pairs])
        sem_2d = np.array([fp.final_2d for fp in session.fiducial_pairs])
        errors = np.array([fp.error for fp in session.fiducial_pairs])
        
        # 3D fluorescence view
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(fluorescence_3d[:, 0], fluorescence_3d[:, 1], fluorescence_3d[:, 2],
                             c=range(len(fluorescence_3d)), cmap='viridis', s=100)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_zlabel('Z (pixels)')
        ax1.set_title('3D Fluorescence Fiducials')
        
        # 2D SEM view
        ax2 = fig.add_subplot(132)
        ax2.scatter(sem_2d[:, 0], sem_2d[:, 1], c=range(len(sem_2d)), cmap='viridis', s=100)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title('2D SEM Fiducials')
        ax2.set_aspect('equal')
        
        # Error visualization
        ax3 = fig.add_subplot(133)
        error_magnitudes = np.linalg.norm(errors, axis=1)
        ax3.hist(error_magnitudes, bins=10, alpha=0.7)
        ax3.set_xlabel('Error magnitude (pixels)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Error Distribution (RMS: {session.transformation.rms_error:.2f})')
        
        # Add transformation info
        transform = session.transformation
        info_text = f"""Transformation Parameters (Empirically Derived):
Rotation: [{transform.rotation_euler[0]:.1f}, {transform.rotation_euler[1]:.1f}, {transform.rotation_euler[2]:.1f}]°
Scale: {transform.scale:.3f}
RMS Error: {transform.rms_error:.2f} pixels
N Fiducials: {len(session.fiducial_pairs)}"""
        
        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def validate_synthetic_data(
        self,
        sessions: List[CorrelationSession],
        real_sessions: List[CorrelationSession] = None
    ) -> Dict[str, float]:
        """Validate synthetic data against real data statistics"""
        
        # Extract statistics from synthetic data
        synthetic_stats = self._extract_session_statistics(sessions)
        
        # Compare with real data if available
        if real_sessions:
            real_stats = self._extract_session_statistics(real_sessions)
            
            validation = {
                'mean_rms_error_diff': abs(synthetic_stats['mean_rms_error'] - real_stats['mean_rms_error']),
                'mean_n_fiducials_diff': abs(synthetic_stats['mean_n_fiducials'] - real_stats['mean_n_fiducials']),
                'rotation_range_overlap': self._check_range_overlap(
                    synthetic_stats['rotation_ranges'], real_stats['rotation_ranges']
                ),
                'scale_range_overlap': self._check_range_overlap(
                    [synthetic_stats['scale_range']], [real_stats['scale_range']]
                )[0]
            }
            
            print("Validation against real data:")
            for key, value in validation.items():
                print(f"  {key}: {value:.3f}")
        else:
            validation = synthetic_stats
            print("Synthetic data statistics:")
            for key, value in validation.items():
                print(f"  {key}: {value}")
        
        return validation
    
    def _extract_session_statistics(self, sessions: List[CorrelationSession]) -> Dict:
        """Extract statistical measures from sessions"""
        rms_errors = [s.transformation.rms_error for s in sessions]
        n_fiducials = [len(s.fiducial_pairs) for s in sessions]
        rotations = [s.transformation.rotation_euler for s in sessions]
        scales = [s.transformation.scale for s in sessions]
        
        # Calculate rotation ranges
        rotation_ranges = []
        for i in range(3):
            rot_values = [r[i] for r in rotations]
            rotation_ranges.append((min(rot_values), max(rot_values)))
        
        return {
            'mean_rms_error': np.mean(rms_errors),
            'std_rms_error': np.std(rms_errors),
            'mean_n_fiducials': np.mean(n_fiducials),
            'rotation_ranges': rotation_ranges,
            'scale_range': (min(scales), max(scales))
        }
    
    def _check_range_overlap(self, ranges1: List[Tuple], ranges2: List[Tuple]) -> List[float]:
        """Check overlap between two sets of ranges"""
        overlaps = []
        for r1, r2 in zip(ranges1, ranges2):
            overlap_start = max(r1[0], r2[0])
            overlap_end = min(r1[1], r2[1])
            overlap = max(0, overlap_end - overlap_start)
            total_range = max(r1[1], r2[1]) - min(r1[0], r2[0])
            overlaps.append(overlap / total_range)
        return overlaps


# Usage example and testing
if __name__ == "__main__":
    print("\n=== Testing Synthetic Data Generator ===")
    
    try:
        # Load real 3DCT sessions
        data_dir = "data"
        print(f"\nLoading real sessions from: {data_dir}")
        parser = ThreeDCTDataParser()
        real_sessions = parser.load_multiple_sessions(data_dir)
        
        if not real_sessions:
            print("✗ No real sessions found. Please check your data directory.")
            exit(1)
        
        print(f"✓ Found {len(real_sessions)} real correlation sessions")
        
        # Create generator
        print("\nCreating synthetic data generator...")
        generator = ThreeDCTSyntheticGenerator.from_real_data(data_dir)
        
        # Generate test session
        print("\nGenerating test session...")
        synthetic_session = generator.generate_correlation_session(seed=42)
        
        print(f"\n=== Test Session Summary ===")
        print(f"Session ID: {synthetic_session.session_id}")
        print(f"Number of fiducials: {len(synthetic_session.fiducial_pairs)}")
        print(f"RMS error: {synthetic_session.transformation.rms_error:.2f} pixels")
        print(f"Rotation: {synthetic_session.transformation.rotation_euler}")
        print(f"Scale: {synthetic_session.transformation.scale:.3f}")
        
        # Visualize
        print("\nVisualizing synthetic session...")
        generator.visualize_synthetic_session(synthetic_session)
        
        # Generate training samples
        print("\nGenerating training samples...")
        training_samples = generator.generate_training_samples(n_samples=5, seed=42)
        
        # Create mixed dataset
        print("\nCreating mixed dataset...")
        mixed_samples = generator.create_mixed_dataset(
            n_synthetic_sessions=10,
            real_sessions=real_sessions,
            synthetic_ratio=0.8,
            seed=42
        )
        
        # Validate
        print("\nValidating synthetic data...")
        synthetic_sessions = [
            generator.generate_correlation_session(seed=100+i) 
            for i in range(5)
        ]
        generator.validate_synthetic_data(synthetic_sessions, real_sessions)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        
        print("\nFalling back to default parameters...")
        # Create generator with default parameters
        default_params = SyntheticParams(
            fluorescence_bounds=((192, 1601), (282, 1229), (47.78, 71.27)),
            sem_bounds=((1076, 2029), (398, 1693)),
            rotation_ranges=((87, 98), (-6, 4), (21, 32)),
            scale_range=(1.008, 1.028),
            translation_origin_ranges=((2200, 2500), (200, 400), (-50, 50)),
            translation_center_ranges=((500, 800), (100, 200), (250, 300)),
            center_point=(828.0, 828.0, 828.0),
            rms_error_range=(0.5, 1.5),
            error_std_per_fiducial=0.8,
            n_fiducials_range=(18, 22),
            n_pois_range=(2, 4)
        )
        
        generator = ThreeDCTSyntheticGenerator(default_params)
        synthetic_session = generator.generate_correlation_session(seed=42)
        print(f"Generated demo session with {len(synthetic_session.fiducial_pairs)} fiducials")
        generator.visualize_synthetic_session(synthetic_session)