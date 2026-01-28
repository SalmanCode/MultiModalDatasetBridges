from dataclasses import dataclass
from typing import Optional

@dataclass
class BridgeConfig:
    bridge_id: str
    bridge_type: str
    span_m: float
    num_spans: int
    total_length_m: float
    width_m: float
    lanes: int
    include_sidewalks: bool
    depth_of_girder: float
    number_of_piers_along_length: int
    number_of_piers_across_width: int
    total_piers: int
    radius_of_pier: float
    pier_type: str
    pier_cap_type: str
    pier_cross_section: str
    top_slab_thk: float = 0.25
    bridge_clearance_height: float = 5.0
    bottom_slab_thk: float = 0.35
    web_thk: float = 0.5
    deck_thickness: float = 0.3
    wing_wall_thickness: float = 0.5
    