import random
from typing import List
from config import BridgeConfig
from dataclasses import asdict
import pandas as pd





BRIDGE_SPECS = {
    "beam_slab": {
        "span": (12.0, 18.0),
        "depth_ratio": (0.06, 0.07),
    },
    "box_girder": {
        "span": (15.0, 37.0),
        "depth_ratio": (0.05, 0.06),
    },
}


def pick_span(bridge_type: str, rng: random.Random, step: int = 5, overhang_m: float = 1.0) -> tuple[float, int, float]:
    spec = BRIDGE_SPECS[bridge_type]
    raw_span = rng.uniform(*spec["span"])
    depth_of_girder = raw_span * rng.uniform(*spec["depth_ratio"])
    num_spans = rng.randint(2, 5)
    total_length = raw_span * num_spans + 2 * overhang_m

    return round(raw_span, 1), num_spans, round(total_length, 1), round(depth_of_girder, 1)

def pick_deck_width(lanes: int, include_sidewalks: bool) -> float:
    lane_width = 3.5  # m
    hard_shoulder = 3.0    # m each side
    hard_strip = 0.5    # m each side
    sidewalk = 1.5 if include_sidewalks else 0.0
    paved_width = lanes * lane_width + 2 * (hard_shoulder + hard_strip)
    return round(paved_width + 2 * sidewalk, 2)

def piers_combination(lanes: int, rng: random.Random, bridge_type: str) -> int:
    #num_of_piers_per_lane = rng.randint(1,2) # this is the number of piers per lane
    num_of_piers_per_lane = 1 # right now we just keep equal to 1 per lane
    radius_of_pier = 0.6 # this is the radius of the pier in meters from oregon state standards
    if bridge_type == "box_girder":
        type_of_pier = rng.choice(["hammer_head"])
    else:
        type_of_pier = rng.choice(["multicolumn"])
    
    pier_cap_type = rng.choice(["prismatic"])
    pier_cross_section = rng.choice(["circular", "rectangular"])
    return num_of_piers_per_lane, radius_of_pier, type_of_pier, pier_cap_type, pier_cross_section


def generate_bridge_configs(count: int, bridge_type: str, seed: int | None = None) -> List[BridgeConfig]:
    rng = random.Random(seed)
    configs: List[BridgeConfig] = []
    step = 5 # this is the step size for span increment. 
    overhang_m = 1.0 # this is the overhang length for the bridge in meters.
    include_sidewalks = True # this is the flag to include sidewalks in the bridge.

    for idx in range(1, count + 1):
        bridge_type_picked = bridge_type or rng.choice(list(BRIDGE_SPECS.keys()))
        lanes = rng.randint(2, 5)
        span, num_spans, total_length, depth_of_girder = pick_span(bridge_type_picked, rng, step, overhang_m)
        width = pick_deck_width(lanes, include_sidewalks)
        number_of_piers_per_lane, radius_of_pier, type_of_pier, pier_cap_type, pier_cross_section = piers_combination(lanes, rng, bridge_type_picked)
        configs.append(BridgeConfig(
            bridge_id=f"bridge_{idx}", 
            bridge_type=bridge_type_picked, 
            span_m=span,
            num_spans=num_spans,
            total_length_m=total_length,
            width_m=width, 
            lanes=lanes,
            include_sidewalks=include_sidewalks,
            depth_of_girder=depth_of_girder,
            number_of_piers_per_lane=number_of_piers_per_lane,
            radius_of_pier=radius_of_pier,
            pier_type=type_of_pier,
            pier_cap_type=pier_cap_type,
            pier_cross_section=pier_cross_section,
            ))
        
    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]

