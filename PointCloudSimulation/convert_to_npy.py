import os
import re
import argparse
import numpy as np
from pathlib import Path
import time




def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc


def convert_bridge_data(input_file, output_dir, add_color_padding=False):
    """Convert bridge point cloud data from HELIOS output to .npy format.
    If add_color_padding is True, output is 6 channels with last 3 as 0,0,0 (shape Nx6).
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    

    
  
        
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        return
        
    print(f"\nConverting {input_file}...")
    print(f"  Source: {input_file}")
    data = np.loadtxt(input_file, delimiter=' ')
    print(f"Shape: {data.shape}, Sample: {data[0][:5]}...")
    
    # Handle different formats
    if data.shape[1] == 11:
        # HELIOS++ format: x y z intensity echo_width return_num num_returns classification gps_time full_wave_idx hitObjectId
        # Extract: x, y, z, intensity, classification
        point_cloud = data[:, [0, 1, 2, 3, 7]]  # x, y, z, intensity, classification
        print(f"Extracted HELIOS++ format: x, y, z, intensity, classification")
    elif data.shape[1] == 6:
        # Standard format: x y z r g b
        point_cloud = data
        point_cloud[:, 3:6] = point_cloud[:, 3:6] / 255.0  # Normalize RGB
        print(f"Using standard x, y, z, r, g, b format")
    else:
        print(f"Warning: Unsupported format with {data.shape[1]} columns")
        return

    # Optionally force 6 channels with last 3 as 0,0,0
    if add_color_padding:
        xyz = point_cloud[:, :3]
        padding = np.zeros((point_cloud.shape[0], 3), dtype=point_cloud.dtype)
        point_cloud = np.concatenate([xyz, padding], axis=1)
        print(f"Added color padding: output shape {point_cloud.shape} (last 3 channels 0,0,0)")
    
    # Ensure we have the right number of points (8192 is standard for PointLLM)
    if len(point_cloud) > 8192:
        # Randomly sample 8192 points
        start_time = time.time()
        print("Randomly sampling 8192 points from ", len(point_cloud))
        indices = np.random.choice(len(point_cloud), 8192, replace=False)
        point_cloud = point_cloud[indices]
        end_time = time.time()
        print(f"Random sampling time: {end_time - start_time} seconds")
    elif len(point_cloud) < 8192:
        print("Upsampling to 8192 points from ", len(point_cloud))
        # Upsample by repeating points
        indices = np.random.choice(len(point_cloud), 8192, replace=True)
        point_cloud = point_cloud[indices]
    
    # Normalize point cloud coordinates
    
    point_cloud = pc_norm(point_cloud)
    
    # Save as .npy file: bridge_<id>_8192.npy
    match = re.search(r"bridge_(\d+)", input_file.stem, re.IGNORECASE)
    output_name = f"bridge_{match.group(1)}_8192.npy" if match else f"{input_file.stem}_8192.npy"
    output_file = os.path.join(output_dir, output_name)
    np.save(output_file, point_cloud.astype(np.float32))
    
    print(f"Saved {output_name} with shape {point_cloud.shape}")
        
    print(f"Conversion complete. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert point cloud to .npy (e.g. 8192 x C).")
    parser.add_argument("--input", "-i", help="Input .xyz file (or path)")
    parser.add_argument("--output", "-o", help="Output directory for .npy files")
    parser.add_argument("--add-color-padding", action="store_true",
                        help="Force 6 channels with last 3 as 0,0,0 (output shape Nx6)")
    args = parser.parse_args()

    input_file = args.input or "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/TLS_5_complete.xyz"
    output_dir = args.output or "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/npy"
    convert_bridge_data(input_file, output_dir, add_color_padding=args.add_color_padding) 
