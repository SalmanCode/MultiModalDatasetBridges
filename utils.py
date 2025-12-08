import random
import logging
import cadquery as cq
from dataclasses import dataclass, asdict
from typing import List, Optional
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



BRIDGE_RANGES: dict[str, tuple[float, float]] = {
    "beam_slab": (20.0, 50.0),
    "box_girder": (50.0, 120.0),
}

def compute_girder_spacing(width_m: float, min_spacing_m: float = 3.5) -> tuple[int, float]:
        
    num_of_girders = max(2, round(width_m / min_spacing_m))
    spacing = width_m / num_of_girders
    return num_of_girders, round(spacing, 1)

def compute_box_girder_spacing(width_m: float, min_spacing_m: float = 5.0) -> int:
    num_of_cells = max(1, round(width_m / (min_spacing_m + 1)) - 1)
    logger.info(f"Width: {width_m}")
    logger.info(f"Min spacing: {min_spacing_m}")
    logger.info(f"Number of cells: {round(width_m / (min_spacing_m + 1)) - 1}")
    return num_of_cells

@dataclass
class BridgeConfig:
    bridge_id: str
    bridge_type: str
    span_m: float
    width_m: float
    deck_m: float
    lanes: int
    include_sidewalks: bool



def pick_span_and_deck(bridge_type: str, rng: random.Random, step: int = 5, overhang_m: float = 1.0) -> tuple[float, float]:
    lower, upper = BRIDGE_RANGES[bridge_type]
    raw_span = rng.uniform(lower, upper)
    span = round(round(raw_span / step) * step, 2)
    deck = round(span + 2 * overhang_m, 2)
    return span, deck

def pick_deck_width(lanes: int, include_sidewalks: bool) -> float:
    lane_width = 3.5  # m
    shoulder = 0.5    # m each side
    sidewalk = 1.5 if include_sidewalks else 0.0
    base = lanes * lane_width + 2 * shoulder
    return round(base + 2 * sidewalk, 2)


def generate_bridge_configs(count: int, step: int, include_sidewalks: bool, seed: int | None = None, overhang_m: float = 1.0) -> List[BridgeConfig]:
    rng = random.Random(seed)
    configs: List[BridgeConfig] = []
    for idx in range(1, count + 1):
        bridge_type = rng.choice(list(BRIDGE_RANGES))
        lanes = rng.randint(3, 5)
        span, deck = pick_span_and_deck(bridge_type, rng, step, overhang_m)
        width = pick_deck_width(lanes, include_sidewalks)
        configs.append(BridgeConfig(bridge_id=f"bridge_{idx}", bridge_type=bridge_type, span_m=span, width_m=width, deck_m=deck, lanes=lanes, include_sidewalks=include_sidewalks))
        
    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]







class BridgeModel:
    def __init__(self, config: BridgeConfig):
        self.config = config

    def make_deck(self) -> cq.Workplane:

        if self.config.bridge_type == "beam_slab":
            return self.make_beam_slab_deck()
        elif self.config.bridge_type == "box_girder":
            return self.make_box_girder_deck()
        else:
            raise ValueError(f"Invalid bridge type: {self.config.bridge_type}")


    def make_box_girder_deck(self) -> cq.Workplane:
        
        
        deck_length = self.config.span_m
        total_width = self.config.width_m

        num_spans = getattr(self.config, "num_spans", 1) if hasattr(self.config, "num_spans") else 1

        effective_span_length = deck_length / num_spans


        # Basic geometric assumptions (meters)
        top_slab_thk = getattr(self.config, "top_slab_thk", 0.25)
        bottom_slab_thk = getattr(self.config, "bottom_slab_thk", 0.15)
        web_thk = getattr(self.config, "web_thk", 0.5)
        box_depth = effective_span_length / 20.0
        inner_cell_gap = 5.0  # desired clear width between webs
        edge_haunch = 0.4     # cantilever each side for parapets

        
        outer_width = inner_cell_gap + web_thk
        num_cells = compute_box_girder_spacing(total_width, inner_cell_gap)

    
       
        top_slab = (
            cq.Workplane("YZ")
            .box(total_width, top_slab_thk, deck_length, centered=(True, False, True)))


        boxes = cq.Workplane("YZ")
        logger.info(f"total width: {total_width}")
        
        for i in range(num_cells):

            box_center_y = - ((num_cells - 1) * outer_width / 2) + outer_width * i 
            
            logger.info(f"Box center y: {box_center_y}")
            

            # outer shell
            outer = cq.Workplane("YZ").rect(outer_width, box_depth, centered=(True, True)).extrude(deck_length).translate((-deck_length / 2, box_center_y, -box_depth/2))
            inner = cq.Workplane("YZ").rect(outer_width - 2 * web_thk, box_depth - top_slab_thk - 0.15 - bottom_slab_thk, centered=(True, True)).extrude(deck_length).translate((-deck_length / 2, box_center_y, -box_depth/2 - 0.15))
            shell = outer.cut(inner)
            
            boxes = boxes.union(shell)

        
        bridge_core = top_slab.union(boxes)
        return bridge_core

    

    def make_beam_slab_deck(self) -> cq.Workplane:
        

        deck_length = self.config.span_m
        width = self.config.width_m
        deck_thickness = getattr(self.config, "deck_thickness", 0.3)

        # Number of spans (default to 1, can be changed for multiplies) 
        num_spans = getattr(self.config, "num_spans", 1) if hasattr(self.config, "num_spans") else 1
        logger.info(f"Number of spans: {num_spans}")

        # Effective span length is the length of the deck that is supported by the piers
        # This will later be changed.
        effective_span_length = deck_length / num_spans


        # Girder height is dependent on effective span length
        girder_height = effective_span_length / 18.0
        girder_thickness = 0.5
        minimum_girder_spacing = 3.5
        chamfer_radius = 0.1
        num_of_girders, girder_distance = compute_girder_spacing(width, minimum_girder_spacing)

        # Ratio of width to girder height. This should be less more than 1/6 ideally.
        ratio = width / girder_height
        logger.info(f"Ratio: {ratio}")

        deck_slab = cq.Workplane("XY").box(deck_length, width, deck_thickness, centered=(True, True, False))
        
        girders = cq.Workplane("YZ")
        for i in range(num_of_girders):
            girder_y_position = -(girder_distance * (num_of_girders - 1) / 2)  + girder_distance * i
            girder = cq.Workplane("YZ").rect(girder_thickness, -girder_height, centered=(True, False)).extrude(deck_length)
            
            girder = girder.translate((-deck_length / 2, girder_y_position , 0)) # remember tranlsation always happens in global coordinates so x, y, z.
            girders = girders.union(girder)

        
        bridge_core = (
            deck_slab.union(girders)
            .faces("-Z")          # bottom faces
            .edges("|X")          # edges parallel to Y (girder webs)
            .chamfer(chamfer_radius)
        )
        
        return bridge_core
        

    
    def build_bridge(self) -> cq.Workplane:
        """Build the complete bridge with optional components"""
        logger.info(f"Building bridge {self.config.bridge_id} with config: {asdict(self.config)}")
        
        # Start with deck
        bridge = self.make_deck()
        
        return bridge