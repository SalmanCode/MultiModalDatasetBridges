from dataclasses import dataclass
from typing import Optional

@dataclass
class BridgeConfig:
    bridge_id: str
    bridge_type: str
    span_m: float
    width_m: float
    deck_m: float
    lanes: int
    include_sidewalks: bool
    top_slab_thk: float = 0.25
    bottom_slab_thk: float = 0.35
    web_thk: float = 0.5
    box_depth: Optional[float] = None
    deck_thickness: float = 0.3
    