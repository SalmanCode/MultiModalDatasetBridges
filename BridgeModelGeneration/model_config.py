from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BridgeConfig:
    bridge_id: str
    bridge_type: str
    span_m: float
    num_spans: int
    inner_deck_length_m: float
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
    road_template: str
    bridge_clearance_height: float = 5.0
    deck_thickness: float = 0.3
    wing_wall_thickness: float = 0.5
    wing_wall_extension_m: float = 0.0
    # Which real-bridge family this config was drawn from. Generator bookkeeping.
    family: str = ""
    # all spans from abutment to abutment
    span_layout_m: List[float] = field(default_factory=list)
    # "wingwall", "spill_through", "prismatic", or "" = pick from num_spans.
    abutment_style: str = ""
    # Horizontal : vertical run of the spill-through slope (real 1.5 - 2.5).
    abutment_slope: float = 2.0
    