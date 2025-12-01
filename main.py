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
    parser.add_argument("--step", type=int, help="Step size for the bridge span", default=5)
    parser.add_argument("--missing_components", action="store_true", help="Include missing components")
    parser.add_argument("--type", default="beam_and_slab", help="Type of bridge to generate (default: beam_and_slab)")
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = BridgePipeline()
    bridge_configs, config_json = pipeline.generate_bridges(args.num_bridges, args.step, args.missing_components)
    

    with open("bridge_summary.json", "w") as f:
        json.dump(config_json, f, indent=2)
    


if __name__ == "__main__":
    main()