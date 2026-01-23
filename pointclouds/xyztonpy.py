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


def convert_bridge_data():
    """Convert bridge point cloud data from HELIOS output to .npy format"""
    
    # Configuration
    helios_output = r"H:\Datasets\syntehtic_data\cad_query\helios\output"
    output_dir = r"H:\Datasets\syntehtic_data\cad_query\pointclouds\PointCloudsNPY"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TLS_bridge_X folders
    bridge_folders = [d for d in os.listdir(helios_output) if d.startswith('TLS_bridge_')]
    bridge_folders.sort()
    
    print(f"Found {len(bridge_folders)} bridge folders in HELIOS output")
    
    for bridge_folder in bridge_folders:
        # Extract bridge ID (e.g., TLS_bridge_1 -> bridge_1)
        bridge_id = bridge_folder.replace('TLS_', '')
        
        # Path to bridge folder
        bridge_path = os.path.join(helios_output, bridge_folder)
        
        # Find the latest timestamp folder
        timestamp_folders = [d for d in os.listdir(bridge_path) if os.path.isdir(os.path.join(bridge_path, d))]
        if not timestamp_folders:
            print(f"Warning: No timestamp folders found in {bridge_folder}")
            continue
        
        # Sort by modification time and get latest
        timestamp_folders.sort(key=lambda x: os.path.getmtime(os.path.join(bridge_path, x)), reverse=True)
        latest_folder = timestamp_folders[0]
        
        # Full path to the complete point cloud file
        bridge_file = os.path.join(bridge_path, latest_folder, f"{bridge_id}_complete.xyz")
        
        if not os.path.exists(bridge_file):
            print(f"Warning: {bridge_file} not found, skipping...")
            continue
            
        print(f"\nConverting {bridge_id}...")
        print(f"  Source: {bridge_folder}/{latest_folder}")
        print(f"  File: {bridge_id}_complete.xyz")
        data = np.loadtxt(bridge_file, delimiter=' ')
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
            continue
        
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
        output_file = os.path.join(output_dir, f"{bridge_id}.npy")
        np.save(output_file, point_cloud.astype(np.float32))
        
        print(f"Saved {bridge_id} with shape {point_cloud.shape}")
            
        # except Exception as e:
        #     print(f"Error processing {bridge_id}: {e}")
        #     continue
    
    print(f"Conversion complete. Files saved to {output_dir}")

if __name__ == "__main__":
    convert_bridge_data() 