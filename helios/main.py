import json
import argparse
from pathlib import Path

from scanner_positions import calculate_scanner_positions
from create_survey_xml import create_survey_xml
from create_scene_xml import create_scene_xml
from run_simulation import run_helios_simulation
from semantic_segmentation import semantic_segmentation


def bridge_pipeline(run_simulation=False, num_bridges=None, run_segmentation=False):
    """Main pipeline to generate HELIOS survey and scene files for bridges.
    
    Args:
        run_simulation: If True, run HELIOS simulations after creating files
        num_bridges: Number of bridges to process (None = all bridges)
        run_segmentation: If True, run semantic segmentation after simulations
    """
    # Paths
    base_dir = Path(__file__).parent.parent
    bridge_summary_path = base_dir / "Generated_Bridges" / "bridge_summary.json"
    helios_dir = base_dir / "helios"
    surveys_dir = helios_dir / "data" / "surveys"
    scenes_dir = helios_dir / "data" / "scenes"
    point_clouds_dir = helios_dir / "output"
    
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
        
        # Run simulation if requested
        if run_simulation:
            print(f"  Running simulation...")
            run_helios_simulation(survey_path)
        
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
        
        # Run semantic segmentation if requested
        if run_segmentation:
            print(f"  Running semantic segmentation...")
            
            # Find the latest timestamped folder in the output directory
            bridge_output_dir = point_clouds_dir / f"TLS_{bridge_id}"
            if bridge_output_dir.exists():
                # Get all timestamped folders and sort to find the latest
                timestamp_folders = sorted([d for d in bridge_output_dir.iterdir() if d.is_dir()], 
                                         key=lambda x: x.stat().st_mtime, reverse=True)
                if timestamp_folders:
                    latest_scan_dir = timestamp_folders[0]
                    print(f"    Processing scan from: {latest_scan_dir.name}")
                    semantic_segmentation(str(latest_scan_dir), bridge_id)
                else:
                    print(f"    Warning: No scan folders found in {bridge_output_dir}")
            else:
                print(f"    Warning: Output directory not found: {bridge_output_dir}")
        
        print(f"Completed {bridge_id}\n")

    
    # Export scanner information to JSON
    scanner_info_path = helios_dir / "scanner_positions.json"
    with open(scanner_info_path, 'w') as f:
        json.dump(scanner_info, f, indent=2)
    print(f"Scanner info exported to: {scanner_info_path}")
    
    if run_simulation:
        print(f"\nAll simulations completed!")
    else:
        print(f"\nAll survey and scene files created successfully!")
        print(f"To run simulations, use: python main.py --run-simulation")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate HELIOS survey and scene files for bridges')
    parser.add_argument('--run-simulation', action='store_true', 
                        help='Run HELIOS simulations after creating survey files')
    parser.add_argument('--num-bridges', type=int, default=None,
                        help='Number of bridges to process (default: all)')
    parser.add_argument('--semantic-segmentation', action='store_true',
                        help='Run semantic segmentation after creating survey files')
    
    args = parser.parse_args()
    bridge_pipeline(run_simulation=args.run_simulation, 
                   num_bridges=args.num_bridges,
                   run_segmentation=args.semantic_segmentation)
