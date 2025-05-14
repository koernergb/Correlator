# src/data_analysis/tdct_data_parser.py
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path

@dataclass
class TransformationParams:
    """3DCT transformation parameters"""
    rotation_euler: np.ndarray  # [phi, psi, theta] in degrees
    scale: float
    translation_origin: np.ndarray  # Translation for rotation around [0,0,0]
    translation_center: np.ndarray  # Translation for rotation around center
    center_point: np.ndarray  # Rotation center [x, y, z]
    rms_error: float
    optimization_successful: bool

@dataclass
class FiducialPair:
    """A verified fiducial correspondence from manual 3DCT correlation"""
    initial_3d: np.ndarray  # [x, y, z] in fluorescence (ground truth input)
    final_2d: np.ndarray    # [x, y] in SEM/FIB (ground truth output)
    transformed_3d: np.ndarray  # Transformed fluorescence coordinates
    error: np.ndarray       # [dx, dy] residual error
    index: int              # Fiducial index

@dataclass 
class POI:
    """Point of Interest (non-fiducial features)"""
    initial_3d: np.ndarray   # [x, y, z] in fluorescence
    correlated_2d: np.ndarray  # [x, y] in SEM/FIB (with z component)
    distance_px: np.ndarray    # Distance from SEM center in pixels
    distance_um: np.ndarray    # Distance from SEM center in micrometers

@dataclass
class CorrelationSession:
    """Complete 3DCT correlation session data"""
    session_id: str
    machine_info: str
    date: str
    transformation: TransformationParams
    fiducial_pairs: List[FiducialPair]  # Verified correspondences (GROUND TRUTH)
    pois: List[POI]
    csv_coordinates: Optional[np.ndarray]  # Optional: preliminary detections
    microscope_center: np.ndarray    # SEM/FIB image center
    pixel_size_um: Optional[float]   # Pixel size in micrometers

class ThreeDCTDataParser:
    """Parser for 3DCT correlation output files"""
    
    def __init__(self, base_path: str = "data"):
        print(f"\nInitializing ThreeDCTDataParser with base path: {base_path}")
        self.base_path = Path(base_path)
    
    def load_session(self, txt_path: str, csv_path: Optional[str] = None) -> CorrelationSession:
        """Load a correlation session from TXT file (primary) and optional CSV"""
        print(f"\n=== Loading Session ===")
        print(f"TXT file: {txt_path}")
        if csv_path:
            print(f"CSV file: {csv_path}")
        
        # Parse the main correlation results from TXT
        print("\nParsing correlation TXT file...")
        correlation_data = self.parse_correlation_txt(txt_path)
        print("✓ TXT file parsed successfully")
        
        # Optionally load CSV coordinates (if provided)
        csv_coordinates = None
        if csv_path and os.path.exists(csv_path):
            try:
                print("\nLoading CSV coordinates...")
                csv_coordinates = self.parse_csv_coordinates(csv_path)
                print(f"✓ CSV coordinates loaded: {len(csv_coordinates)} points")
            except Exception as e:
                print(f"✗ Error loading CSV file {csv_path}: {str(e)}")
                print("Stack trace:")
                import traceback
                traceback.print_exc()
        
        # Create session ID from filename
        session_id = Path(txt_path).stem
        print(f"\nCreating session with ID: {session_id}")
        
        return CorrelationSession(
            session_id=session_id,
            machine_info=correlation_data['machine_info'],
            date=correlation_data['date'],
            transformation=correlation_data['transformation'],
            fiducial_pairs=correlation_data['fiducial_pairs'],
            pois=correlation_data['pois'],
            csv_coordinates=csv_coordinates,
            microscope_center=correlation_data['microscope_center'],
            pixel_size_um=correlation_data['pixel_size_um']
        )
    
    def parse_csv_coordinates(self, csv_path: str) -> np.ndarray:
        """Parse coordinates from CSV file (treat as preliminary/supplementary)"""
        print(f"\nParsing CSV file: {csv_path}")
        try:
            # Try reading as CSV
            print("Attempting to read as CSV...")
            df = pd.read_csv(csv_path, header=None)
            coords = df.iloc[:, :2].values  # Take first two columns as X, Y
            print(f"✓ Successfully read CSV: {len(coords)} coordinates")
            return coords
        except:
            # Fallback: read as space/tab separated
            try:
                print("CSV read failed, trying space/tab separated format...")
                coords = np.loadtxt(csv_path, usecols=(0, 1))
                print(f"✓ Successfully read space/tab separated: {len(coords)} coordinates")
                return coords
            except:
                # Last resort: try reading all columns and pick first 2
                print("Space/tab separated read failed, trying raw data read...")
                data = np.loadtxt(csv_path)
                if data.ndim == 1:
                    # Single row
                    coords = data[:2].reshape(1, 2)
                else:
                    coords = data[:, :2]
                print(f"✓ Successfully read raw data: {len(coords)} coordinates")
                return coords
    
    def parse_correlation_txt(self, txt_path: str) -> Dict:
        """Parse 3DCT correlation output text file - main ground truth source"""
        print(f"\nParsing correlation TXT file: {txt_path}")
        try:
            with open(txt_path, 'r') as f:
                content = f.read()
            print("✓ File read successfully")
            
            # Extract metadata
            print("\nExtracting metadata...")
            machine_info = self._extract_machine_info(content)
            date = self._extract_date(content)
            print(f"Machine info: {machine_info}")
            print(f"Date: {date}")
            
            # Extract transformation parameters
            print("\nExtracting transformation parameters...")
            transformation = self._parse_transformation_params(content)
            print(f"Rotation: {transformation.rotation_euler}")
            print(f"Scale: {transformation.scale}")
            print(f"RMS error: {transformation.rms_error}")
            
            # Extract verified fiducial correspondences
            print("\nExtracting fiducial pairs...")
            fiducial_pairs = self._parse_fiducial_pairs(content)
            print(f"Found {len(fiducial_pairs)} fiducial pairs")
            
            # Extract POIs
            print("\nExtracting points of interest...")
            pois = self._parse_pois(content)
            print(f"Found {len(pois)} points of interest")
            
            # Extract microscope center
            print("\nExtracting microscope center...")
            microscope_center = self._parse_microscope_center(content)
            print(f"Microscope center: {microscope_center}")
            
            # Extract pixel size
            print("\nExtracting pixel size...")
            pixel_size = self._parse_pixel_size(content)
            print(f"Pixel size: {pixel_size} um")
            
            return {
                'machine_info': machine_info,
                'date': date,
                'transformation': transformation,
                'fiducial_pairs': fiducial_pairs,
                'pois': pois,
                'microscope_center': microscope_center,
                'pixel_size_um': pixel_size
            }
            
        except Exception as e:
            print(f"\n✗ Error parsing TXT file: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_machine_info(self, content: str) -> str:
        """Extract machine information"""
        match = re.search(r"# Machine: (.+)", content)
        return match.group(1).strip() if match else ""
    
    def _extract_date(self, content: str) -> str:
        """Extract experiment date"""
        match = re.search(r"# Date: (.+)", content)
        return match.group(1).strip() if match else ""
    
    def _parse_transformation_params(self, content: str) -> TransformationParams:
        """Extract transformation parameters from text"""
        # Rotation (Euler angles)
        rotation_match = re.search(r"rotation \(Euler phi, psi, theta\): \[([^\]]+)\]", content)
        if rotation_match:
            rotation_str = rotation_match.group(1)
            rotation = np.array([float(x.strip()) for x in rotation_str.split(',')])
        else:
            rotation = np.zeros(3)
        
        # Scale
        scale_match = re.search(r"scale =\s*([0-9.]+)", content)
        scale = float(scale_match.group(1)) if scale_match else 1.0
        
        # Translation for origin
        trans_origin_pattern = r"translation for rotation around \[0,0,0\] = \[([^\]]+)\]"
        trans_origin_match = re.search(trans_origin_pattern, content)
        if trans_origin_match:
            trans_origin_str = trans_origin_match.group(1)
            trans_origin = np.array([float(x.strip()) for x in trans_origin_str.split(',')])
        else:
            trans_origin = np.zeros(3)
        
        # Translation for center and center point
        center_pattern = r"translation for rotation around \[([^\]]+)\] = \[([^\]]+)\]"
        center_match = re.search(center_pattern, content)
        if center_match:
            center_str = center_match.group(1)
            trans_center_str = center_match.group(2)
            center_point = np.array([float(x.strip()) for x in center_str.split(',')])
            trans_center = np.array([float(x.strip()) for x in trans_center_str.split(',')])
        else:
            center_point = np.zeros(3)
            trans_center = np.zeros(3)
        
        # RMS error
        rms_match = re.search(r"rms error \(in 2d pixels\) =\s*([0-9.]+)", content)
        rms_error = float(rms_match.group(1)) if rms_match else 0.0
        
        # Optimization success
        opt_match = re.search(r"optimization successful", content)
        optimization_successful = opt_match is not None
        
        return TransformationParams(
            rotation_euler=rotation,
            scale=scale,
            translation_origin=trans_origin,
            translation_center=trans_center,
            center_point=center_point,
            rms_error=rms_error,
            optimization_successful=optimization_successful
        )
    
    def _parse_fiducial_pairs(self, content: str) -> List[FiducialPair]:
        """Extract verified fiducial correspondences from correlation table"""
        # Find the fiducial correlation table
        table_start = content.find("Initial (3D) markers")
        table_end = content.find("# Correlation of 3D spots")
        
        if table_start == -1:
            # Alternative header format
            table_start = content.find("# Transformation of initial (3D) markers")
            if table_start != -1:
                # Skip the header lines
                lines_after = content[table_start:].split('\n')
                for i, line in enumerate(lines_after):
                    if "Initial (3D)" in line and "Final (2D)" in line:
                        table_start = table_start + len('\n'.join(lines_after[:i+1]))
                        break
        
        if table_start == -1:
            print("Warning: Could not find fiducial correlation table")
            return []
        
        if table_end == -1:
            table_end = len(content)
        
        table_section = content[table_start:table_end]
        lines = table_section.split('\n')
        
        fiducial_pairs = []
        index = 0
        
        for line in lines:
            # Skip headers, comments, and empty lines
            if (line.startswith('#') or 
                'Initial' in line or 'Final' in line or 'Transformed' in line or
                'markers' in line or not line.strip()):
                continue
            
            # Parse data line: should have at least 10 numeric values
            parts = line.split()
            if len(parts) >= 10:
                try:
                    # Parse coordinates
                    initial_3d = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                    transformed_3d = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                    final_2d = np.array([float(parts[6]), float(parts[7])])
                    error = np.array([float(parts[8]), float(parts[9])])
                    
                    # Create fiducial pair
                    fiducial_pairs.append(FiducialPair(
                        initial_3d=initial_3d,
                        final_2d=final_2d,
                        transformed_3d=transformed_3d,
                        error=error,
                        index=index
                    ))
                    index += 1
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
        
        return fiducial_pairs
    
    def _parse_pois(self, content: str) -> List[POI]:
        """Extract points of interest from text"""
        poi_start = content.find("# Correlation of 3D spots (POIs) to 2D")
        distance_start = content.find("# POI distance from the center")
        
        if poi_start == -1:
            return []
        
        # Parse correlation section
        section_end = distance_start if distance_start != -1 else len(content)
        poi_section = content[poi_start:section_end]
        lines = poi_section.split('\n')
        
        pois = []
        for line in lines:
            if line.strip() and not line.startswith('#') and 'Spots' not in line and 'Correlated' not in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        initial_3d = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                        correlated_2d = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                        
                        # Initialize distance arrays (will be filled if distance data exists)
                        distance_px = np.zeros(2)
                        distance_um = np.zeros(2)
                        
                        pois.append(POI(
                            initial_3d=initial_3d,
                            correlated_2d=correlated_2d,
                            distance_px=distance_px,
                            distance_um=distance_um
                        ))
                    except (ValueError, IndexError):
                        continue
        
        # Parse distance information if available
        if distance_start != -1:
            self._parse_poi_distances(content[distance_start:], pois)
        
        return pois
    
    def _parse_poi_distances(self, distance_section: str, pois: List[POI]):
        """Parse POI distance information and update POI objects"""
        lines = distance_section.split('\n')
        poi_index = 0
        
        for line in lines:
            if line.strip() and not line.startswith('#') and 'Distance' not in line:
                parts = line.split()
                if len(parts) >= 4 and poi_index < len(pois):
                    try:
                        distance_px = np.array([float(parts[0]), float(parts[1])])
                        distance_um = np.array([float(parts[2]), float(parts[3])])
                        
                        pois[poi_index].distance_px = distance_px
                        pois[poi_index].distance_um = distance_um
                        poi_index += 1
                    except (ValueError, IndexError):
                        continue
    
    def _parse_microscope_center(self, content: str) -> np.ndarray:
        """Extract SEM/FIB image center coordinates"""
        center_match = re.search(r"center is at x/y = ([0-9.]+)/([0-9.]+)", content)
        if center_match:
            return np.array([float(center_match.group(1)), float(center_match.group(2))])
        return np.array([0.0, 0.0])
    
    def _parse_pixel_size(self, content: str) -> Optional[float]:
        """Extract pixel size in micrometers"""
        pixel_match = re.search(r"pixel size: ([0-9.]+) um", content)
        if pixel_match:
            return float(pixel_match.group(1))
        # Check for 'nan' case
        if 'pixel size: nan um' in content:
            return None
        return None
    
    def convert_to_e3nn_format(self, transform: TransformationParams) -> np.ndarray:
        """Convert transformation parameters to E3NN format
        
        Args:
            transform: Transformation parameters from 3DCT
            
        Returns:
            Array of parameters in E3NN format:
            - First 9 elements: flattened rotation matrix
            - Next 1 element: scale
            - Next 3 elements: translation
            - Last 3 elements: center point
        """
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
        
        # Combined rotation: R = R_z * R_y * R_x (same order as 3DCT)
        R = R_z @ R_y @ R_x
        
        # Combine all parameters
        params = np.concatenate([
            R.flatten(),  # 9 parameters for rotation
            [transform.scale],  # 1 parameter for scale
            transform.translation_center,  # 3 parameters for translation
            transform.center_point  # 3 parameters for center point
        ])
        
        return params

    def extract_training_pairs(self, session: CorrelationSession) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract verified (3D fluorescence, 2D SEM) coordinate pairs and transformation parameters for training
        
        Returns:
            List of tuples: (3D_fluorescence_coords, 2D_SEM_coords, transform_params)
        """
        training_pairs = []
        
        # Convert transformation parameters to E3NN format
        transform_params = self.convert_to_e3nn_format(session.transformation)
        
        for fiducial in session.fiducial_pairs:
            # Ground truth input: 3D fluorescence coordinates (with Z!)
            fluorescence_3d = fiducial.initial_3d  # [x, y, z]
            
            # Ground truth output: 2D SEM coordinates (manually verified)
            sem_2d = fiducial.final_2d  # [x, y]
            
            training_pairs.append((fluorescence_3d, sem_2d, transform_params))
        
        return training_pairs
    
    def extract_training_data_with_metadata(self, session: CorrelationSession) -> Dict:
        """Extract training data with additional metadata for analysis"""
        training_pairs = self.extract_training_pairs(session)
        
        # Separate into arrays for easier manipulation
        fluorescence_coords = np.array([pair[0] for pair in training_pairs])  # (N, 3)
        sem_coords = np.array([pair[1] for pair in training_pairs])  # (N, 2)
        
        # Extract errors for quality assessment
        errors = np.array([fid.error for fid in session.fiducial_pairs])  # (N, 2)
        error_magnitudes = np.linalg.norm(errors, axis=1)  # (N,)
        
        return {
            'fluorescence_3d': fluorescence_coords,
            'sem_2d': sem_coords,
            'errors': errors,
            'error_magnitudes': error_magnitudes,
            'rms_error': session.transformation.rms_error,
            'n_fiducials': len(training_pairs),
            'transformation_params': session.transformation,
            'session_id': session.session_id
        }
    
    def get_transformation_matrix_homogeneous(self, session: CorrelationSession) -> np.ndarray:
        """Convert 3DCT transformation parameters to 4x4 homogeneous matrix"""
        transform = session.transformation
        
        # Convert Euler angles to rotation matrix
        phi, psi, theta = np.radians(transform.rotation_euler)
        
        # Rotation matrices for each axis (order: X, Y, Z)
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
        
        # Combined rotation: R = R_z * R_y * R_x (same order as 3DCT)
        R = R_z @ R_y @ R_x
        
        # Create 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = transform.scale * R  # Apply scale to rotation
        T[:3, 3] = transform.translation_center  # Translation vector
        
        return T
    
    def load_multiple_sessions(self) -> List[CorrelationSession]:
        """Load all correlation sessions from a directory"""
        print(f"\n=== Loading Multiple Sessions ===")
        print(f"Data directory: {self.base_path}")
        
        data_path = self.base_path
        sessions = []
        
        # Find all correlation txt files
        print("\nSearching for correlation files...")
        txt_files = list(data_path.glob("*correlation*.txt"))
        print(f"Found {len(txt_files)} correlation TXT files")
        
        for txt_file in txt_files:
            print(f"\nProcessing file: {txt_file.name}")
            
            # Look for corresponding CSV file
            base_name = txt_file.stem
            possible_csv_names = [
                f"predictions.csv",  # Common name
                f"{base_name.replace('_correlation', '')}.csv",
                f"{base_name}.csv"
            ]
            
            csv_file = None
            for csv_name in possible_csv_names:
                candidate = data_path / csv_name
                if candidate.exists():
                    csv_file = str(candidate)
                    print(f"Found matching CSV: {csv_name}")
                    break
            
            try:
                session = self.load_session(str(txt_file), csv_file)
                sessions.append(session)
                print(f"✓ Successfully loaded session: {session.session_id}")
                print(f"  - {len(session.fiducial_pairs)} fiducial pairs")
                print(f"  - RMS error: {session.transformation.rms_error:.2f} pixels")
                if csv_file:
                    print(f"  - CSV data: {len(session.csv_coordinates)} coordinates")
            except Exception as e:
                print(f"✗ Error loading {txt_file.name}: {str(e)}")
                print("Stack trace:")
                import traceback
                traceback.print_exc()
        
        print(f"\n=== Loading Complete ===")
        print(f"Successfully loaded {len(sessions)} sessions")
        return sessions
    
    def validate_session_data(self, session: CorrelationSession) -> Dict[str, bool]:
        """Validate the parsed session data for consistency"""
        print(f"\n=== Validating Session: {session.session_id} ===")
        
        validation = {
            'has_fiducials': len(session.fiducial_pairs) > 0,
            'optimization_successful': session.transformation.optimization_successful,
            'reasonable_rms': 0.1 <= session.transformation.rms_error <= 5.0,
            'has_3d_coords': False,
            'has_2d_coords': False,
            'coordinate_bounds_ok': True
        }
        
        print("\nChecking fiducials...")
        print(f"Has fiducials: {validation['has_fiducials']}")
        print(f"Optimization successful: {validation['optimization_successful']}")
        print(f"RMS error: {session.transformation.rms_error:.2f} (reasonable: {validation['reasonable_rms']})")
        
        if validation['has_fiducials']:
            print("\nValidating coordinates...")
            for i, fid in enumerate(session.fiducial_pairs):
                # Check 3D coordinates
                if len(fid.initial_3d) == 3 and np.all(np.isfinite(fid.initial_3d)):
                    validation['has_3d_coords'] = True
                
                # Check 2D coordinates  
                if len(fid.final_2d) == 2 and np.all(np.isfinite(fid.final_2d)):
                    validation['has_2d_coords'] = True
                
                # Basic bounds checking
                if (np.any(fid.initial_3d < 0) or np.any(fid.initial_3d > 10000) or
                    np.any(fid.final_2d < 0) or np.any(fid.final_2d > 10000)):
                    validation['coordinate_bounds_ok'] = False
                    print(f"✗ Fiducial {i} has out-of-bounds coordinates")
        
        validation['overall_valid'] = all([
            validation['has_fiducials'],
            validation['has_3d_coords'], 
            validation['has_2d_coords'],
            validation['coordinate_bounds_ok']
        ])
        
        print("\nValidation Results:")
        for key, value in validation.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        return validation


# Usage example and testing
if __name__ == "__main__":
    print("\n=== Testing ThreeDCTDataParser ===")
    
    # Initialize parser
    parser = ThreeDCTDataParser()
    
    # Test with your data files
    txt_path = "data/2023_embo_clem_material/3DCT/data/2023-02-16_14-43-14_correlation.txt"
    csv_path = "data/2023_embo_clem_material/3DCT/data/predictions.csv"  # Optional
    
    try:
        # Load session
        print("\nLoading test session...")
        session = parser.load_session(txt_path, csv_path)
        
        print("\n=== Session Summary ===")
        print(f"Session ID: {session.session_id}")
        print(f"Date: {session.date}")
        print(f"Number of verified fiducial pairs: {len(session.fiducial_pairs)}")
        print(f"Number of POIs: {len(session.pois)}")
        print(f"RMS error: {session.transformation.rms_error:.2f} pixels")
        print(f"Optimization successful: {session.transformation.optimization_successful}")
        
        # Validate session data
        print("\nValidating session data...")
        validation = parser.validate_session_data(session)
        
        # Extract training pairs (main output)
        print("\nExtracting training pairs...")
        training_pairs = parser.extract_training_pairs(session)
        print(f"Number of training pairs: {len(training_pairs)}")
        
        # Show first few examples
        print("\nFirst 3 training pairs:")
        for i, (fluor_3d, sem_2d, _) in enumerate(training_pairs[:3]):
            print(f"  Pair {i+1}:")
            print(f"    3D Fluorescence: ({fluor_3d[0]:.1f}, {fluor_3d[1]:.1f}, {fluor_3d[2]:.1f})")
            print(f"    2D SEM:          ({sem_2d[0]:.1f}, {sem_2d[1]:.1f})")
        
        # Get training data with metadata
        print("\nExtracting training data with metadata...")
        training_data = parser.extract_training_data_with_metadata(session)
        print(f"\n=== Training Statistics ===")
        print(f"Average error magnitude: {np.mean(training_data['error_magnitudes']):.2f} pixels")
        print(f"Max error magnitude: {np.max(training_data['error_magnitudes']):.2f} pixels")
        print(f"Min error magnitude: {np.min(training_data['error_magnitudes']):.2f} pixels")
        
        # Get transformation matrix
        print("\nComputing transformation matrix...")
        transform_matrix = parser.get_transformation_matrix_homogeneous(session)
        print(f"\n=== Transformation Matrix ===")
        print("4x4 Homogeneous transformation matrix:")
        print(transform_matrix)
        
        # Show coordinate ranges
        fluor_coords = training_data['fluorescence_3d']
        sem_coords = training_data['sem_2d']
        print(f"\n=== Coordinate Ranges ===")
        print(f"3D Fluorescence:")
        print(f"  X: {fluor_coords[:, 0].min():.1f} to {fluor_coords[:, 0].max():.1f}")
        print(f"  Y: {fluor_coords[:, 1].min():.1f} to {fluor_coords[:, 1].max():.1f}")
        print(f"  Z: {fluor_coords[:, 2].min():.1f} to {fluor_coords[:, 2].max():.1f}")
        print(f"2D SEM:")
        print(f"  X: {sem_coords[:, 0].min():.1f} to {sem_coords[:, 0].max():.1f}")
        print(f"  Y: {sem_coords[:, 1].min():.1f} to {sem_coords[:, 1].max():.1f}")
        
        # Check if CSV matches TXT (if available)
        if session.csv_coordinates is not None:
            print(f"\n=== CSV vs TXT Comparison ===")
            print(f"CSV coordinates: {len(session.csv_coordinates)}")
            print(f"TXT fiducials: {len(session.fiducial_pairs)}")
            
            # Compare 2D projections if possible
            if len(session.csv_coordinates) > 0:
                csv_first_few = session.csv_coordinates[:3]
                txt_first_few = fluor_coords[:3, :2]  # Project to 2D
                print("First 3 CSV coordinates:")
                for i, coord in enumerate(csv_first_few):
                    print(f"  CSV {i+1}: ({coord[0]:.1f}, {coord[1]:.1f})")
                print("First 3 TXT 3D coordinates (X,Y only):")
                for i, coord in enumerate(txt_first_few):
                    print(f"  TXT {i+1}: ({coord[0]:.1f}, {coord[1]:.1f})")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
    
    # Test loading multiple sessions
    print(f"\n=== Testing Multiple Session Loading ===")
    try:
        sessions = parser.load_multiple_sessions()
        print(f"Found {len(sessions)} valid correlation sessions")
        
        if len(sessions) > 1:
            # Analyze consistency across sessions
            rms_errors = [s.transformation.rms_error for s in sessions]
            n_fiducials = [len(s.fiducial_pairs) for s in sessions]
            
            print(f"\n=== Session Statistics ===")
            print(f"RMS error range: {min(rms_errors):.2f} - {max(rms_errors):.2f} pixels")
            print(f"Fiducial count range: {min(n_fiducials)} - {max(n_fiducials)}")
    except Exception as e:
        print(f"\n✗ Error loading multiple sessions: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
