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

from utils import BridgeConfig, generate_bridge_configs, configs_to_records, BridgeModel


logger = logging.getLogger(__name__)


class BridgePipeline:
    """High-level pipeline orchestrating parameter generation and exports."""

    def __init__(self, base_dir: str = "."):
        
        # set the directories
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "Generated_Bridges"
        self.bridge_objects_dir = self.output_dir / "BridgeObjects"
        self.metadata_file = self.output_dir / "bridge_metadata.xlsx"

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.bridge_objects_dir.mkdir(exist_ok=True)

        # Track generated bridges
        self.bridge_metadata: List[BridgeConfig] = []

    def generate_bridges(self, num_bridges: int, step: int, include_sidewalks: bool, seed = None, overhang_m: float = 1.0) -> List[BridgeConfig]:
        """Create bridge configs and keep them in-memory."""

        configs = generate_bridge_configs(count=num_bridges, step=step, include_sidewalks=include_sidewalks, seed=seed, overhang_m=overhang_m)
        self.bridge_metadata = configs
        logger.info("Generated %d bridge configs", len(configs))

        config_json = configs_to_records(configs)

        for config in configs:
            bridge_model = BridgeModel(config)
            bridge = bridge_model.build_bridge()
            stl_file = self.bridge_objects_dir / f"{config.bridge_id}.stl"
            cq.exporters.export(bridge, str(stl_file))

        return configs, config_json

    

    