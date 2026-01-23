import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_npy(npy_file):
    """Visualize a point cloud from .npy file"""
    
    # Load the point cloud
    pc = np.load(npy_file)
    print(f"Point cloud shape: {pc.shape}")
    
    # Extract coordinates
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D scatter with intensity (if available)
    ax1 = fig.add_subplot(131, projection='3d')
    if pc.shape[1] >= 4:
        intensity = pc[:, 3]
        scatter = ax1.scatter(x, y, z, c=intensity, cmap='viridis', s=1)
        plt.colorbar(scatter, ax=ax1, label='Intensity')
    else:
        ax1.scatter(x, y, z, s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View (colored by intensity)')
    
    # Plot 2: Top view with classification (if available)
    ax2 = fig.add_subplot(132)
    if pc.shape[1] >= 5:
        classification = pc[:, 4]
        scatter = ax2.scatter(x, y, c=classification, cmap='tab10', s=1)
        plt.colorbar(scatter, ax=ax2, label='Class')
        ax2.set_title('Top View (colored by class)')
    else:
        ax2.scatter(x, y, s=1)
        ax2.set_title('Top View')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    
    # Plot 3: Side view
    ax3 = fig.add_subplot(133)
    if pc.shape[1] >= 5:
        scatter = ax3.scatter(x, z, c=classification, cmap='tab10', s=1)
        plt.colorbar(scatter, ax=ax3, label='Class')
        ax3.set_title('Side View (colored by class)')
    else:
        ax3.scatter(x, z, s=1)
        ax3.set_title('Side View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()


def visualize_xyz(xyz_file, max_points=50000):
    """Visualize a point cloud from .xyz file (HELIOS++ format)"""
    
    print(f"Loading {xyz_file}...")
    # Load only first max_points for faster visualization
    data = np.loadtxt(xyz_file, delimiter=' ', max_rows=max_points)
    print(f"Loaded {len(data)} points")
    
    # Extract coordinates and classification
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D view with intensity
    ax1 = fig.add_subplot(131, projection='3d')
    if data.shape[1] >= 4:
        intensity = data[:, 3]
        scatter = ax1.scatter(x, y, z, c=intensity, cmap='viridis', s=0.5, alpha=0.5)
        plt.colorbar(scatter, ax=ax1, label='Intensity')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View (intensity)')
    
    # Plot 2: Top view with classification
    ax2 = fig.add_subplot(132)
    if data.shape[1] >= 8:
        classification = data[:, 7]
        scatter = ax2.scatter(x, y, c=classification, cmap='tab10', s=0.5, alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax2, label='Classification')
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['approach', 'back_wall', 'deck', 'piers', 'railings', 'wings'])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (classification)')
    ax2.axis('equal')
    
    # Plot 3: Side view
    ax3 = fig.add_subplot(133)
    if data.shape[1] >= 8:
        scatter = ax3.scatter(x, z, c=classification, cmap='tab10', s=0.5, alpha=0.5)
        plt.colorbar(scatter, ax=ax3, label='Classification')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (classification)')
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_pointcloud.py <file.npy>")
        print("  python visualize_pointcloud.py <file.xyz>")
        print("\nExample:")
        print("  python visualize_pointcloud.py PointCloudsNPY/bridge_1.npy")
        print("  python visualize_pointcloud.py PointCloudsXYZ/bridge_1/bridge_1.xyz")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if file_path.endswith('.npy'):
        visualize_npy(file_path)
    elif file_path.endswith('.xyz'):
        visualize_xyz(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        print("Supported formats: .npy, .xyz")
