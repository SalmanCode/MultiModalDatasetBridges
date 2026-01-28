import logging
from pathlib import Path
from typing import List, Optional
import cadquery as cq
import open3d as o3d
import pandas as pd
from dataclasses import asdict
from datetime import datetime
import time
import random
import os
import json
from .model_config import BridgeConfig
from .param_gen import generate_bridge_configs, configs_to_records
from .bridge_model import BridgeModel



logger = logging.getLogger(__name__)


class BridgePipeline:
    """High-level pipeline orchestrating parameter generation and exports."""

    def __init__(self, base_dir: str = "."):
        
        # set the directories
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "Dataset"
        self.bridge_objects_dir = self.output_dir / "BridgeModels"
        self.metadata_file = self.output_dir / "bridge_metadata.xlsx"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.bridge_objects_dir.mkdir(exist_ok=True)

        # Track generated bridges
        self.bridge_metadata: List[BridgeConfig] = []

    def generate_bridges(self, num_bridges: int, bridge_type: str, include_components: bool = False, seed: int | None = None) -> List[BridgeConfig]:
        """Create bridge configs and keep them in-memory."""

        # First we generate the bridge configs from config.py
        configs = generate_bridge_configs(count=num_bridges, bridge_type=bridge_type, seed=seed)
        self.bridge_metadata = configs
        logger.info("Generated %d bridge configs", len(configs))

        # Then we convert the configs to records

        config_json = configs_to_records(configs)
        with open(self.output_dir / "bridge_summary.json", "w") as f:
            json.dump(config_json, f, indent=2)

        for config in configs:
            # Then we build the bridge model from geometry.bridge_model.py
            bridge_model = BridgeModel(config)

            bridge = bridge_model.build_bridge(with_components=False) # this is building the bridge model from geometry.bridge_model.py
            stl_file = self.bridge_objects_dir / f"{config.bridge_id}.stl"
            cq.exporters.export(bridge, str(stl_file))
            mesh = o3d.io.read_triangle_mesh(str(stl_file))
            obj_file = self.bridge_objects_dir / f"{config.bridge_id}.obj"
            if len(mesh.vertices) > 0:
                o3d.io.write_triangle_mesh(str(obj_file), mesh)
                # Clean up STL file
                stl_file.unlink()
                logger.info(f"Successfully saved bridge {config.bridge_id} to {obj_file}")
            
            if include_components:
                components, bridge = bridge_model.build_bridge(with_components=True)
                os.makedirs(self.bridge_objects_dir / f"{config.bridge_id}", exist_ok=True) #making a directory for the bridge id
                stl_file = self.bridge_objects_dir / f"{config.bridge_id}.stl"

                for name, component in components.items():
                    stl_file = self.bridge_objects_dir / f"{config.bridge_id}" / f"{name}.stl"
                    cq.exporters.export(component, str(stl_file))

                    mesh = o3d.io.read_triangle_mesh(str(stl_file))
                    obj_file = self.bridge_objects_dir / f"{config.bridge_id}" / f"{name}.obj"
                    if len(mesh.vertices) > 0:
                        o3d.io.write_triangle_mesh(str(obj_file), mesh)
                        # Clean up STL file
                        stl_file.unlink()
                        logger.info(f"Successfully saved bridge {config.bridge_id} to {obj_file}")
                    else:
                        logger.error(f"Failed to generate valid mesh for bridge {config.bridge_id}")
                        return
            
                

        return configs, config_json

    

    