import json
import argparse
from pathlib import Path
import os
from collections import defaultdict

from .scanner_positions import calculate_scanner_positions
from .create_survey_xml import create_survey_xml
from .create_scene_xml import create_scene_xml
from .semantic_segmentation import semantic_segmentation
from .convert_to_npy import convert_bridge_data


def pointcloud_complete_pipeline(run_simulation=False, num_bridges=None, run_segmentation=False, convert_to_npy=False):
    """Main pipeline to generate point cloud complete dataset.
    This pipeline will generate the point cloud complete dataset including the raw point clouds, segmented point clouds, and merged point clouds.
    It will also convert the point clouds to NPY format if requested.
    Args:
        run_simulation: If True, run HELIOS simulations after creating files
        num_bridges: Number of bridges to process (None = all bridges).
        run_segmentation: If True, run semantic segmentation after simulations
        convert_to_npy: If True, convert the point clouds to NPY format
    """
    # Paths
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "Dataset"
    bridge_summary_path = dataset_dir / "bridge_summary.json"
    helios_dir = base_dir / "PointCloudSimulation"
    surveys_dir = helios_dir / "data" / "surveys"
    scenes_dir = helios_dir / "data" / "scenes"
    
    
    # Dataset output paths
    scan_output_dir = dataset_dir / "PointCloudScans"
    scan_legs_output_dir = scan_output_dir / "scan_legs"
    segmented_output_dir = scan_output_dir / "segmented"
    merged_output_dir = scan_output_dir / "merged"
    
    
    # Create directories if they don't exist
    surveys_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    # Read bridge summary
    with open(bridge_summary_path, 'r') as f:
        bridges = json.load(f)
    
    # Limit number of bridges if specified
    if num_bridges is not None:
        bridges = bridges[:num_bridges]
        print(f"Processing {len(bridges)} of {len(json.load(open(bridge_summary_path)))} bridges\n")
    else:
        print(f"Found {len(bridges)} bridges to process\n")
    
    # Store scanner info for export
    scanner_info = []
    
    # Process each bridge
    for bridge in bridges:
        bridge_id = bridge['bridge_id']
        print(f"Processing {bridge_id}...")
        
        # Calculate scanner positions
        positions = calculate_scanner_positions(bridge)
        print(f"  Scanner positions calculated:")
        for leg_name, pos in positions.items():
            print(f"    {leg_name}: x={pos['x']:.1f}, y={pos['y']:.1f}, z={pos['z']:.1f}")
        
        # Create survey XML
        survey_path = surveys_dir / f"TLS_{bridge_id}_survey.xml"
        create_survey_xml(bridge, positions, survey_path)
        
        # Create scene XML
        scene_path = scenes_dir / f"TLS_{bridge_id}_scene.xml"
        create_scene_xml(bridge, scene_path)
        
        # Store bridge info for export
        scanner_info.append({
            'bridge_id': bridge_id,
            'dimensions': {
                'width_m': bridge['width_m'],
                'length_m': bridge['total_length_m']
            },
            'scanner_positions': {
                leg_name: {
                    'x': round(pos['x'], 2),
                    'y': round(pos['y'], 2),
                    'z': round(pos['z'], 2)
                }
                for leg_name, pos in positions.items()
            }
        })
        
        
        
        
        # Run simulation if requested
        if run_simulation:
            scan_legs_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running simulation...")
            cmd = f"helios {survey_path} --output {scan_legs_output_dir} -vt" #-vt means verbose output only time and errors will be reported
            print(f"Running simulation: {cmd}")
            os.system(cmd)
            print(f"Saving raw point clouds with timestamped folders in {scan_legs_output_dir} with each legs.")
            
            # Merging the leg scans into a single file
            print(f"Merging the leg scans into a single file...")


            # now we go to each tls bridge folder and pick the last simulation run.

            # this is the directory where the leg scans are saved
            scan_legs_bridge_dir = scan_legs_output_dir / f"TLS_{bridge_id}"
            if scan_legs_bridge_dir.exists():
                # Get all timestamped folders and sort to find the latest
                timestamp_folders = sorted([d for d in scan_legs_bridge_dir.iterdir() if d.is_dir()], 
                                         key=lambda x: x.stat().st_mtime, reverse=True)
                
                latest_scan_dir = timestamp_folders[0]
                print(f"Processing legs scan from: {scan_legs_bridge_dir / latest_scan_dir.name}")
                    
                #reading all leg xyz files from the latest scan directory.
                xyz_files = [
                        os.path.join(latest_scan_dir, fname)
                        for fname in os.listdir(latest_scan_dir)
                        if fname.endswith(".xyz")
                    ]
                
                all_lines = []
                component_files = defaultdict(list)
                
                for leg_file in xyz_files:
                    with open(leg_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue

                            # Collect line for the merged output
                            all_lines.append(line)

                            # Split the line by whitespace
                            parts = line.split()
                            
                            # Get the 9th column (index 8)
                            component_id = int(parts[8])
                            component_files[component_id].append(line)
                
                # Write merged .xyz file containing all scans
                merged_bridge_dir = merged_output_dir / f"TLS_{bridge_id}"
                os.makedirs(merged_bridge_dir, exist_ok=True)
                merged_output_file = merged_bridge_dir / f"{bridge_id}_complete.xyz"
                with open(merged_output_file, 'w') as f_merged:
                    f_merged.write('\n'.join(all_lines) + '\n')
                    print(f"  ✓ Merged all scans into {bridge_id}_complete.xyz ({len(all_lines):,} points)")
        
        

                # Run semantic segmentation if requested
                if run_segmentation:

                    print(f"  Running semantic segmentation...")            
                    # Creating the segmented output directory for the bridge
                    segmented_output_dir.mkdir(parents=True, exist_ok=True)
                    segmented_bridge_dir = segmented_output_dir / f"TLS_{bridge_id}"
                    os.makedirs(segmented_bridge_dir, exist_ok=True)
                    # Run segmentation with Dataset output directories
                    semantic_segmentation(component_files, segmented_bridge_dir)
                    print(f"  ✓ Semantic segmentation completed")
                else:
                    print(f"No semantic segmentation requested")

            

               
            # Convert to NPY format if requested
                if convert_to_npy:
                    print(f"\n{'='*70}")
                    print(f"Converting point clouds to NPY format...")
                    print(f"{'='*70}\n")

                    npy_output_dir = scan_output_dir / "npy"
                    os.makedirs(npy_output_dir, exist_ok=True)
                    convert_bridge_data(merged_output_file, npy_output_dir, add_color_padding=True)
                    
            else:
    
                print(f"Warning: Output directory not found: {scan_legs_bridge_dir}")
            
            print(f"Completed {bridge_id}\n")

    
    # Export scanner information to JSON
    dataset_dir = base_dir / "Dataset" / "PointCloudScans"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    scanner_info_path = dataset_dir / "scanner_positions.json"
    with open(scanner_info_path, 'w') as f:
        json.dump(scanner_info, f, indent=2)
    print(f"Scanner info exported to: {scanner_info_path}")
    
    
    


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HELIOS survey and scene files for bridges')
    parser.add_argument('--run_simulation', action='store_true', 
                        help='Run HELIOS simulations after creating survey files')
    parser.add_argument('--num_bridges', type=int, default=None,
                        help='Number of bridges to process (default: all)')
    parser.add_argument('--semantic_segmentation', action='store_true',
                        help='Run semantic segmentation after simulations')
    parser.add_argument('--convert_to_npy', action='store_true',
                        help='Convert point clouds to NPY format after simulations')
    
    args = parser.parse_args()
    pointcloud_complete_pipeline(run_simulation=args.run_simulation, 
                   num_bridges=args.num_bridges,
                   run_segmentation=args.semantic_segmentation,
                   convert_to_npy=args.convert_to_npy)
