# src/data_analysis/debug_rgb_images.py
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def load_and_fix_3dct_data(base_path="data/3D_correlation_test_dataset"):
    """Load 3DCT data with proper RGB handling"""
    
    # Load coordinate files
    ib_coords = np.loadtxt(f"{base_path}/IB_coordinates.txt")
    lm_coords = np.loadtxt(f"{base_path}/LM_coordinates.txt") 
    
    # Load images
    ib_image = tifffile.imread(f"{base_path}/IB_image.tif")
    lm_stack = tifffile.imread(f"{base_path}/LM_image_stack_reslized.tif")
    
    print("=== Image Analysis ===")
    print(f"IB image shape: {ib_image.shape}")
    print(f"IB image dtype: {ib_image.dtype}")
    print(f"IB image range: {ib_image.min():.1f} to {ib_image.max():.1f}")
    print(f"LM stack shape: {lm_stack.shape}")
    print(f"LM stack dtype: {lm_stack.dtype}")
    print(f"LM stack range: {lm_stack.min():.1f} to {lm_stack.max():.1f}")
    
    return ib_coords, lm_coords, ib_image, lm_stack

def normalize_image(image):
    """Normalize image to 0-1 range for proper display"""
    if len(image.shape) == 3:  # RGB image
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            channel = image[:, :, i].astype(np.float32)
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                normalized[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
            else:
                normalized[:, :, i] = channel
        return normalized
    else:  # Grayscale
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            return (image.astype(np.float32) - image_min) / (image_max - image_min)
        else:
            return image.astype(np.float32)

def convert_rgb_to_grayscale(rgb_image):
    """Convert RGB image to grayscale using standard weights"""
    if len(rgb_image.shape) == 3:
        # Standard RGB to grayscale conversion
        weights = np.array([0.299, 0.587, 0.114])
        grayscale = np.dot(rgb_image, weights)
        return grayscale
    return rgb_image

def debug_coordinate_system(ib_coords, lm_coords, ib_image, lm_stack):
    """Debug the coordinate system with proper image handling"""
    
    # Normalize images for display
    ib_normalized = normalize_image(ib_image)
    ib_gray = convert_rgb_to_grayscale(ib_normalized)
    
    # For LM, create max projection
    lm_max_proj = np.max(lm_stack, axis=0)
    lm_normalized = normalize_image(lm_max_proj)
    
    # Test different coordinate interpretations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # IB image tests - show both RGB and grayscale
    ax = axes[0, 0]
    ax.imshow(ib_normalized)
    ax.scatter(ib_coords[:, 0], ib_coords[:, 1], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('IB RGB with coords (X, Y)')
    
    ax = axes[0, 1]
    ax.imshow(ib_gray, cmap='gray')
    ax.scatter(ib_coords[:, 0], ib_coords[:, 1], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('IB Grayscale with coords (X, Y)')
    
    ax = axes[0, 2]
    ax.imshow(ib_gray, cmap='gray')
    ax.scatter(ib_coords[:, 1], ib_coords[:, 0], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('IB Grayscale with coords (Y, X)')
    
    ax = axes[0, 3]
    ax.imshow(ib_gray, cmap='gray')
    # Try flipping Y coordinate
    ax.scatter(ib_coords[:, 0], ib_image.shape[0] - ib_coords[:, 1], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('IB Grayscale with Y flipped')
    
    # LM image tests
    ax = axes[1, 0]
    ax.imshow(lm_normalized, cmap='gray')
    ax.scatter(lm_coords[:, 0], lm_coords[:, 1], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('LM with coords (X, Y)')
    
    ax = axes[1, 1]
    ax.imshow(lm_normalized, cmap='gray')
    ax.scatter(lm_coords[:, 1], lm_coords[:, 0], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('LM with coords (Y, X)')
    
    ax = axes[1, 2]
    ax.imshow(lm_normalized, cmap='gray')
    ax.scatter(lm_coords[:, 0], lm_normalized.shape[0] - lm_coords[:, 1], c='red', s=100, marker='+', linewidth=3)
    ax.set_title('LM with Y flipped')
    
    # Try different coordinate scaling
    ax = axes[1, 3]
    ax.imshow(lm_normalized, cmap='gray')
    # Check if coordinates need scaling
    scale_x = lm_normalized.shape[1] / 1000  # Example scaling
    scale_y = lm_normalized.shape[0] / 1000
    ax.scatter(lm_coords[:, 0] * scale_x, lm_coords[:, 1] * scale_y, c='red', s=100, marker='+', linewidth=3)
    ax.set_title('LM with scaled coords')
    
    plt.tight_layout()
    plt.show()

def inspect_patches(ib_coords, ib_image):
    """Inspect patches around coordinates to see what's there"""
    
    ib_gray = convert_rgb_to_grayscale(normalize_image(ib_image))
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    patch_size = 50
    
    for i, coord in enumerate(ib_coords[:9]):
        if i >= len(axes):
            break
            
        ax = axes[i]
        x, y = int(coord[0]), int(coord[1])
        
        # Extract patch
        if (x > patch_size and x < ib_gray.shape[1] - patch_size and
            y > patch_size and y < ib_gray.shape[0] - patch_size):
            patch = ib_gray[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
            ax.imshow(patch, cmap='gray')
            ax.axhline(y=patch_size, color='red', linestyle='--')
            ax.axvline(x=patch_size, color='red', linestyle='--')
            ax.set_title(f'Patch {i+1} at ({x}, {y})')
        else:
            ax.text(0.5, 0.5, 'Out of bounds', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Patch {i+1} at ({x}, {y}) - OOB')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data with proper handling
    ib_coords, lm_coords, ib_image, lm_stack = load_and_fix_3dct_data()
    
    # Debug coordinate system
    debug_coordinate_system(ib_coords, lm_coords, ib_image, lm_stack)
    
    # Inspect what's actually at the coordinate locations
    inspect_patches(ib_coords, ib_image)