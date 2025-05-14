# src/data_analysis/examine_3dct_data.py
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_3dct_test_data(base_path="data/3D_correlation_test_dataset"):
    """Load and examine the 3DCT test dataset"""
    
    # Load coordinate files
    ib_coords = np.loadtxt(f"{base_path}/IB_coordinates.txt")
    lm_coords = np.loadtxt(f"{base_path}/LM_coordinates.txt") 
    lm_coords_poi = np.loadtxt(f"{base_path}/LM_coordinates_withPOI.txt")
    
    # Load images
    ib_image = tifffile.imread(f"{base_path}/IB_image.tif")
    lm_stack = tifffile.imread(f"{base_path}/LM_image_stack_reslized.tif")
    
    return {
        'ib_coords': ib_coords,
        'lm_coords': lm_coords,
        'lm_coords_poi': lm_coords_poi,
        'ib_image': ib_image,
        'lm_stack': lm_stack
    }

def analyze_fiducial_characteristics(data):
    """Analyze characteristics of real fiducials"""
    
    print("=== 3DCT Test Dataset Analysis ===")
    print(f"IB image shape: {data['ib_image'].shape}")
    print(f"LM stack shape: {data['lm_stack'].shape}")
    print(f"IB coordinates shape: {data['ib_coords'].shape}")
    print(f"LM coordinates shape: {data['lm_coords'].shape}")
    print(f"LM POI coordinates shape: {data['lm_coords_poi'].shape}")
    
    # Examine coordinate formats
    print(f"\nIB coordinates sample:\n{data['ib_coords'][:3]}")
    print(f"\nLM coordinates sample:\n{data['lm_coords'][:3]}")
    
    # Check if coordinates match (should have same number of points)
    print(f"\nNumber of IB fiducials: {len(data['ib_coords'])}")
    print(f"Number of LM fiducials: {len(data['lm_coords'])}")
    
    return data

def visualize_fiducials(data):
    """Visualize fiducials in both modalities"""
    fig = plt.figure(figsize=(15, 10))
    
    # IB image with fiducials (2D)
    ax1 = fig.add_subplot(221)
    ax1.imshow(data['ib_image'], cmap='gray')
    # For 2D image, plot (x,y) directly as (col,row)
    ax1.scatter(data['ib_coords'][:, 0], data['ib_coords'][:, 1], 
                c='red', s=50, alpha=0.8, marker='+')
    ax1.set_title('Ion Beam Image with Fiducials')
    ax1.set_xlabel('X (columns)')
    ax1.set_ylabel('Y (rows)')
    
    # LM maximum projection with fiducials
    ax2 = fig.add_subplot(222)
    lm_max_proj = np.max(data['lm_stack'], axis=0)  # Max projection along z
    ax2.imshow(lm_max_proj, cmap='gray')
    # For 2D projection, plot (x,y) directly
    ax2.scatter(data['lm_coords'][:, 0], data['lm_coords'][:, 1], 
                c='red', s=50, alpha=0.8, marker='+')
    ax2.set_title('LM Maximum Projection with Fiducials')
    ax2.set_xlabel('X (columns)')
    ax2.set_ylabel('Y (rows)')
    
    # 3D view of LM coordinates
    ax3 = fig.add_subplot(223, projection='3d')
    # For 3D plot, use all coordinates (x,y,z)
    ax3.scatter(data['lm_coords'][:, 0], data['lm_coords'][:, 1], 
               data['lm_coords'][:, 2], c='red', s=50)
    ax3.set_title('LM Fiducials in 3D')
    ax3.set_xlabel('X (columns)')
    ax3.set_ylabel('Y (rows)')
    ax3.set_zlabel('Z (slices)')
    
    # Pixel intensity analysis around fiducials
    ax4 = fig.add_subplot(224)
    
    # Extract patches around IB fiducials (using correct y,x indexing)
    patch_size = 20
    ib_patches = []
    for coord in data['ib_coords'][:5]:  # First 5 fiducials
        x, y = int(coord[0]), int(coord[1])
        if (x > patch_size and x < data['ib_image'].shape[1] - patch_size and
            y > patch_size and y < data['ib_image'].shape[0] - patch_size):
            # Use [y,x] indexing for image access
            patch = data['ib_image'][y-patch_size:y+patch_size, 
                                   x-patch_size:x+patch_size]
            ib_patches.append(patch)
    
    if ib_patches:
        avg_patch = np.mean(ib_patches, axis=0)
        im = ax4.imshow(avg_patch, cmap='gray')
        ax4.set_title('Average Fiducial Appearance (IB)')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()

def extract_fiducial_statistics(data):
    """Extract useful statistics from real fiducial data"""
    stats = {}
    
    # Size analysis (estimate fiducial size in pixels)
    ib_image = data['ib_image']
    fiducial_sizes = []
    
    for coord in data['ib_coords']:
        x, y = int(coord[0]), int(coord[1])
        # Extract small patch around fiducial using [y,x] indexing
        patch_size = 10
        if (x > patch_size and x < ib_image.shape[1] - patch_size and
            y > patch_size and y < ib_image.shape[0] - patch_size):
            patch = ib_image[y-patch_size:y+patch_size, 
                           x-patch_size:x+patch_size]
            
            # Find connected component size (simplified)
            center_intensity = patch[patch_size, patch_size]
            # Count pixels within 80% of center intensity
            similar_pixels = np.sum(patch > 0.8 * center_intensity)
            fiducial_sizes.append(similar_pixels)
    
    stats['fiducial_sizes_pixels'] = fiducial_sizes
    stats['avg_fiducial_size'] = np.mean(fiducial_sizes)
    stats['std_fiducial_size'] = np.std(fiducial_sizes)
    
    # Spatial distribution (using x,y coordinates directly)
    ib_coords = data['ib_coords']
    distances = []
    for i in range(len(ib_coords)):
        for j in range(i+1, len(ib_coords)):
            # Calculate 2D distance (ignoring z)
            dist = np.sqrt((ib_coords[i,0] - ib_coords[j,0])**2 + 
                         (ib_coords[i,1] - ib_coords[j,1])**2)
            distances.append(dist)
    
    stats['pairwise_distances'] = distances
    stats['avg_distance'] = np.mean(distances)
    stats['min_distance'] = np.min(distances)
    
    return stats

def calibrate_physics_simulation(data, stats):
    """Use real data statistics to calibrate our physics simulation"""
    print("\n=== Calibration Parameters for Physics Simulation ===")
    
    # Estimate pixel size (need metadata for this)
    # For now, assume typical values
    pixel_size_nm = 10  # nm per pixel (typical for EM)
    
    # Convert fiducial size from pixels to physical units
    avg_size_pixels = stats['avg_fiducial_size']
    estimated_diameter_nm = np.sqrt(avg_size_pixels) * pixel_size_nm
    
    print(f"Average fiducial size: {avg_size_pixels:.1f} pixels²")
    print(f"Estimated diameter: {estimated_diameter_nm:.0f} nm")
    
    # Spatial distribution
    print(f"Average distance: {stats['avg_distance']:.1f} pixels")
    print(f"Minimum distance: {stats['min_distance']:.1f} pixels")
    
    # Image dimensions give us grid size (using correct axis order)
    grid_size_pixels = (data['ib_image'].shape[1], data['ib_image'].shape[0])  # (width, height)
    grid_size_microns = (grid_size_pixels[0] * pixel_size_nm / 1000,
                        grid_size_pixels[1] * pixel_size_nm / 1000)
    
    print(f"Grid size: {grid_size_microns} μm")
    
    return {
        'estimated_bead_diameter_um': estimated_diameter_nm / 1000,
        'grid_size_um': grid_size_microns,
        'pixel_size_nm': pixel_size_nm,
        'min_distance_pixels': stats['min_distance']
    }

# Main analysis
if __name__ == "__main__":
    # Load the real test data
    data = load_3dct_test_data()
    
    # Analyze characteristics
    data = analyze_fiducial_characteristics(data)
    
    # Visualize
    visualize_fiducials(data)
    
    # Extract statistics  
    stats = extract_fiducial_statistics(data)
    
    # Calibrate simulation
    calibration = calibrate_physics_simulation(data, stats)
    
    print(f"\nSuggested simulation parameters:")
    print(f"- Bead diameter: {calibration['estimated_bead_diameter_um']:.3f} μm")
    print(f"- Grid size: {calibration['grid_size_um']}")
    print(f"- Minimum spacing: {calibration['min_distance_pixels']} pixels")