from BridgePipeline import BridgePipeline
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bridge datasets")
    parser.add_argument("num_bridges", type=int, help="Number of bridges to generate", default=20)
    parser.add_argument("--bridge_type", type=str, help="Type of girder to generate (beam_slab, box_girder)", default=None)
    parser.add_argument("--step", type=int, help="Step size for the bridge span", default=5)
    parser.add_argument("--include_sidewalks", action="store_true", help="Include sidewalks")
    parser.add_argument("--missing_components", action="store_true", help="Include missing components")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = BridgePipeline()
    bridge_configs, config_json = pipeline.generate_bridges(args.num_bridges, args.bridge_type, args.step, args.include_sidewalks, seed=None, overhang_m=1.0)
    

    with open("bridge_summary.json", "w") as f:
        json.dump(config_json, f, indent=2)
    


if __name__ == "__main__":
    main()