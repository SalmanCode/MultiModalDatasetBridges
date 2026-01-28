import os
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


def convert_bridge_data(input_file, output_dir):
    """Convert bridge point cloud data from HELIOS output to .npy format"""
    

    
  
        
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
    
    # Save as .npy file
    output_file = os.path.join(output_dir, f"{input_file.stem}.npy")
    np.save(output_file, point_cloud.astype(np.float32))
    
    print(f"Saved {input_file.stem} with shape {point_cloud.shape}")
        
    print(f"Conversion complete. Files saved to {output_dir}")

if __name__ == "__main__":
    input_file = "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/TLS_5_complete.xyz"
    output_dir = "H:/Datasets/syntehtic_data/cad_query/helios/bridge_5/npy"
    convert_bridge_data(input_file, output_dir)
