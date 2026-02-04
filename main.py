#!/usr/bin/env python
"""
Complete Bridge Dataset Pipeline

Generates synthetic bridge models, simulates laser scanning with HELIOS++,
performs semantic segmentation, and converts to NPY format for ML training.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

from BridgeModelGeneration.bridge_pipeline import BridgePipeline
from PointCloudSimulation.run_simulations import pointcloud_complete_pipeline
from PointCloudSimulation.convert_to_npy import convert_bridge_data

BASE_DIR = Path(__file__).parent


def generate_bridges(num_bridges, bridge_type=None, include_components=False):
    """Step 1: Generating 3D bridge models
    This will call the BridgePipeline class to generate the bridge models.
    Use include components to create separate component files (approach_slabs, deck, etc.)."""
    print(f"\n{'='*70}")
    print(f"STEP 1: Generating {num_bridges} bridge models")
    print(f"{'='*70}\n")
    
    try:
        start_time = time.time()
        pipeline = BridgePipeline(base_dir=str(BASE_DIR))
        bridge_configs, config_json = pipeline.generate_bridges(
            num_bridges=num_bridges,
            bridge_type=bridge_type,
            include_components=include_components,
            seed=None
        )
        print(f"\nSuccessfully generated {num_bridges} bridges")
        end_time = time.time()
        print(f"Time taken for generating bridges: {end_time - start_time} seconds")
        return True
    except Exception as e:
        print(f"\nError generating bridges: {e}")
        return False


def run_helios_simulation(num_bridges, run_simulation=True, run_segmentation=False, npy_conversion=False):
    """Step 2: Runing HELIOS++ simulation to generate point clouds"""
    print(f"\n{'='*70}")
    print(f"STEP 2: Running HELIOS++ simulations")
    print(f"{'='*70}\n")
    
    try:
        start_time = time.time()
        pointcloud_complete_pipeline(
            run_simulation=run_simulation,
            num_bridges=num_bridges,
            run_segmentation=run_segmentation,
            convert_to_npy=npy_conversion
        )
        print(f"\nSuccessfully completed simulation pipeline")
        end_time = time.time()
        print(f"Time taken for complete simulation pipeline: {end_time - start_time} seconds")
        return True
    except Exception as e:
        print(f"\nError in HELIOS simulation. {e}")
        return False



def verify_output():
    """Verify that outputs were created"""
    print(f"\n{'='*70}")
    print("VERIFICATION: Checking outputs")
    print(f"{'='*70}\n")
    
    dataset_dir = BASE_DIR / "Dataset"
    
    # Check bridge models
    bridge_models = dataset_dir / "BridgeModels"
    if bridge_models.exists():
        obj_files = list(bridge_models.glob("bridge_*/"))
        print(f"Bridge models: {len(obj_files)} bridges found")
    else:
        print(f"Bridge models folder not found")
    
    # Check raw scans
    raw_scans = dataset_dir / "PointCloudScans" / "scan_legs"
    if raw_scans.exists():
        scan_folders = list(raw_scans.glob("TLS_*/"))
        print(f"Raw point clouds: {len(scan_folders)} bridge scans found")
    else:
        print(f"Scan legs folder not found")
    
    # Check segmented scans
    segmented_scans = dataset_dir / "PointCloudScans" / "segmented"
    if segmented_scans.exists():
        seg_folders = list(segmented_scans.glob("TLS_*/"))
        print(f"Segmented point clouds: {len(seg_folders)} found")
    else:
        print(f"Segmented scans folder not found")
    
    # Check NPY files
    npy_folder = dataset_dir / "PointCloudScans" / "npy"
    if npy_folder.exists():
        npy_files = list(npy_folder.glob("*.npy"))
        print(f"NPY point clouds: {len(npy_files)} files found")
    else:
        print(f"NPY folder not found")
    
    # Check bridge summary
    summary_file = dataset_dir / "bridge_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            bridges = json.load(f)
        print(f"Bridge metadata: {len(bridges)} bridges documented")
    else:
        print(f"Bridge summary not found")
    
    # Check scanner positions
    scanner_pos = dataset_dir / "PointCloudScans" / "scanner_positions.json"
    if scanner_pos.exists():
        print(f"Scanner positions: documented")
    else:
        print(f"Scanner positions not found")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline for synthetic bridge dataset generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with 5 bridges
    python main.py--num-bridges 5

    # Complete pipeline with components, simulation, and segmentation and npy conversion 
    python main.py --num-bridges 10 --include-components --run-simulation --semantic-segmentation --npy-conversion

    """
    )
    # Bridge generation options
    parser.add_argument('--num-bridges', type=int, required=True,
                        help='Number of bridges to generate')
    parser.add_argument('--bridge-type', type=str, choices=['box_girder', 'beam_slab'],
                        help='Type of bridge to generate (default: mixed)')
    parser.add_argument('--include-components', action='store_true',
                        help='Generate separate component files (approach_slabs, deck, etc.)')
    # HELIOS simulation options
    parser.add_argument('--run-simulation', action='store_true',
                        help='Run HELIOS simulation')
    parser.add_argument('--semantic-segmentation', action='store_true',
                        help='Perform semantic segmentation on point clouds')
    # Conversion options
    parser.add_argument('--npy-conversion', action='store_true',
                        help='Convert point clouds to NPY format')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.semantic_segmentation and not args.run_simulation:
        print("Warning: --semantic-segmentation requires --run-simulation")
        args.run_simulation = True
    
    if args.npy_conversion and not args.run_simulation:
        print("Warning: --npy-conversion requires --run-simulation")
        args.run_simulation = True
    
    # Print configuration
    print("\n" + "="*70)
    print("SYNTHETIC BRIDGE DATASET PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Bridges: {args.num_bridges}")
    print(f"  • Bridge type: {args.bridge_type or 'mixed'}")
    print(f"  • Include components: {args.include_components}")
    print(f"  • Run simulation: {args.run_simulation}")
    print(f"  • Semantic segmentation: {args.semantic_segmentation}")
    print(f"  • Convert to NPY: {args.npy_conversion}")
    print()
    
    # Execute pipeline
    success = True
    
    # Step 1: Generate bridges
    if not generate_bridges(args.num_bridges, args.bridge_type, args.include_components):
        print("\n❌ Pipeline failed at bridge generation step")
        sys.exit(1)
    
    # Step 2: Run HELIOS simulation (if requested)
    if args.run_simulation:
        if not run_helios_simulation(args.num_bridges, args.run_simulation, args.semantic_segmentation, args.npy_conversion):
            print("\n❌ Pipeline failed at HELIOS simulation step")
            sys.exit(1)
        
    # Verification
    verify_output()
    


if __name__ == "__main__":
    main()
