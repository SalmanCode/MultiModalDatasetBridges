"""
Point Cloud Visualization Tool
Visualizes .npy point cloud files using Open3D
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import sys


def load_npy_pointcloud(filepath):
    """
    Load point cloud from .npy file
    
    Args:
        filepath: Path to .npy file
        
    Returns:
        numpy array of shape (N, 3) or (N, 6) where columns are [x, y, z] or [x, y, z, r, g, b]
    """
    try:
        data = np.load(filepath)
        print(f"Loaded point cloud with shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        if len(data.shape) == 1:
            # If 1D, try to reshape to (N, 3) or (N, 6)
            if data.shape[0] % 3 == 0:
                data = data.reshape(-1, 3)
            elif data.shape[0] % 6 == 0:
                data = data.reshape(-1, 6)
                
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def create_open3d_pointcloud(points, colors=None):
    """
    Create Open3D point cloud object
    
    Args:
        points: numpy array of shape (N, 3) with xyz coordinates
        colors: optional numpy array of shape (N, 3) with RGB values (0-255 or 0-1)
        
    Returns:
        open3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Normalize colors to 0-1 range if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default gray color
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    return pcd


def visualize_pointcloud(filepath, show_normals=False, estimate_normals=False, 
                         point_size=2.0, background_color="white"):
    """
    Visualize point cloud from .npy file
    
    Args:
        filepath: Path to .npy file
        show_normals: Whether to display normals
        estimate_normals: Whether to estimate normals if not present
        point_size: Size of points in visualization
        background_color: Background color ("white" or "black")
    """
    # Load data
    print(f"Loading point cloud from: {filepath}")
    data = load_npy_pointcloud(filepath)
    
    # Extract xyz coordinates
    if data.shape[1] >= 3:
        points = data[:, :3]
        print(f"Number of points: {points.shape[0]}")
        print(f"Point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    else:
        print("Error: Data must have at least 3 columns (x, y, z)")
        sys.exit(1)
    
    # Extract colors if present
    colors = None
    if data.shape[1] >= 6:
        colors = data[:, 3:6]
        print("Colors detected in data")
    
    # Create Open3D point cloud
    pcd = create_open3d_pointcloud(points, colors)
    
    # Estimate normals if requested
    if estimate_normals or show_normals:
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
    
    # Set up visualization
    print("\nVisualization Controls:")
    print("  - Mouse left button: Rotate")
    print("  - Mouse right button: Pan")
    print("  - Mouse wheel: Zoom")
    print("  - Press 'H' for help")
    print("  - Press 'Q' or ESC to quit")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud Viewer - {Path(filepath).name}")
    vis.add_geometry(pcd)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    if background_color.lower() == "white":
        render_option.background_color = np.array([1, 1, 1])
    else:
        render_option.background_color = np.array([0, 0, 0])
    
    if show_normals:
        render_option.point_show_normal = True
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Visualize point cloud from .npy file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "filepath",
        type=str,
        nargs='?',
        default="../Dataset/PointCloudScans/npy/bridge_1_complete.npy",
        help="Path to .npy point cloud file"
    )
    
    parser.add_argument(
        "--normals",
        action="store_true",
        help="Estimate and display normals"
    )
    
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Size of points in visualization"
    )
    
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Background color"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        print(f"Absolute path: {filepath.absolute()}")
        sys.exit(1)
    
    # Visualize
    visualize_pointcloud(
        filepath=filepath,
        show_normals=args.normals,
        estimate_normals=args.normals,
        point_size=args.point_size,
        background_color=args.background
    )


if __name__ == "__main__":
    main()
