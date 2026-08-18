import random
from typing import List
from .model_config import BridgeConfig
from dataclasses import asdict
import pandas as pd
from random import choice

    

ROAD_TEMPLATES = {
    "RQ9":  {"lanes": 2, "lane_w": 3.00, "shoulder": 0.0, "edge": 0.50, "cap_range": (0.75, 2.0)},
    "RQ11": {"lanes": 2, "lane_w": 3.50, "shoulder": 0.0, "edge": 0.50, "cap_range": (0.75, 2.0)},
    "RQ11_5": {"lanes": 2, "lane_w": 3.50, "shoulder": 0.0, "edge": 1.00, "cap_range": (0.75, 2.0)},
    "RQ15_5": {"lanes": 3, "lane_w": 3.50, "shoulder": 0.0, "edge": 1.00, "cap_range": (0.75, 2.0)},
    "RQ31_one_deck": {"lanes": 2, "lane_w": 3.75, "shoulder": 3.0, "edge": 0.75, "cap_range": (0.75, 2.0)},
    "RQ36_one_deck": {"lanes": 3, "lane_w": 3.75, "shoulder": 2.5, "edge": 0.75, "cap_range": (0.75, 2.0)},
}

# currently we are not using the road template weights: all are equally sampled.

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


def pick_span(bridge_type: str, rng: random.Random, overhang_m: float = 1.0, wing_wall_extension_m: float = 0.0) -> tuple[float, int, float]:
    """Pick a span for a bridge."""
    spec = BRIDGE_SPECS[bridge_type]
    raw_span = rng.uniform(*spec["span"])
    depth_of_girder = raw_span * rng.uniform(*spec["depth_ratio"])
    num_spans = rng.randint(1, 5)
    inner_deck_length = raw_span * num_spans + 2 * overhang_m
    total_length = inner_deck_length + 2 * wing_wall_extension_m
    #Changed: We no longer use the step size for span increment.

    return round(raw_span, 1), num_spans, round(inner_deck_length, 1), round(total_length, 1), round(depth_of_girder, 1)

def pick_deck_width(rng: random.Random):
    """Sample width from a German ROAD_TEMPLATES entry."""
    names = list(ROAD_TEMPLATES.keys())
    name = rng.choices(names, k=1)[0]
    t = ROAD_TEMPLATES[name]

    cap = rng.uniform(*t["cap_range"])
    roadway_width = t["lanes"] * t["lane_w"] + 2 * (t["shoulder"] + t["edge"])
    width = round(roadway_width + 2 * cap, 2)

    include_sidewalks = cap > 1.0

    return width, t["lanes"], name, include_sidewalks



def piers_combination(lanes: int, rng: random.Random, bridge_type: str, width_m: float, depth_of_girder: float, num_spans: int) -> int:
    
    num_of_piers_per_lane = 1 # right now we just keep equal to 1 per lane
    radius_of_pier = 0.6 # this is the radius of the pier in meters from oregon state standards
    if bridge_type == "box_girder":
        type_of_pier = rng.choice(["hammer_head"])
    else:
        type_of_pier = rng.choice(["multicolumn"])
    
    pier_cap_type = rng.choice(["prismatic"])
    pier_cross_section = rng.choice(["circular", "rectangular"])

    num_of_piers_along_length = num_spans - 1
    if num_of_piers_along_length == 0:
        num_of_piers_across_width = 0
        pier_cross_section = None
        pier_cap_type = None
        type_of_pier = None

    else:
        if type_of_pier == "hammer_head":
            # depth/width bands from literature (notion): wider shallow decks → 2
            # hammer heads; deeper relative girders → 1. ratio > 0.20 also → 1.
            ratio = round(depth_of_girder / width_m, 2)
            if ratio < 0.16:
                num_of_piers_across_width = 2
            else:
                num_of_piers_across_width = 1
        elif type_of_pier == "multicolumn":
            num_of_piers_across_width = num_of_piers_per_lane * lanes
        else:
            num_of_piers_across_width = num_of_piers_per_lane * lanes

    #this whole logic should be expanded more with respect to zhang
    
    return num_of_piers_along_length, num_of_piers_across_width, radius_of_pier, type_of_pier, pier_cap_type, pier_cross_section


def generate_bridge_configs(count: int, bridge_type: str, seed: int | None = None) -> List[BridgeConfig]:
    rng = random.Random(seed)
    configs: List[BridgeConfig] = [] 
    overhang_m = 1.0 # this is the overhang length for the bridge in meters.
    wing_wall_extension_m = 4.0 # this is the extension length for the wing walls in meters.
    #include_sidewalks = True # this is the flag to include sidewalks in the bridge.

    for idx in range(1, count + 1):
        bridge_type_picked = bridge_type or rng.choice(list(BRIDGE_SPECS.keys()))
        span, num_spans, inner_deck_length, total_length, depth_of_girder = pick_span(bridge_type_picked, rng, overhang_m, wing_wall_extension_m)
        width, lanes, road_template, include_sidewalks = pick_deck_width(rng)
        number_of_piers_along_length, number_of_piers_across_width, radius_of_pier, type_of_pier, pier_cap_type, pier_cross_section = piers_combination(lanes, rng, bridge_type_picked, width, depth_of_girder, num_spans)
        total_piers = number_of_piers_along_length * number_of_piers_across_width
        configs.append(BridgeConfig(
            bridge_id=f"bridge_{idx}", 
            bridge_type=bridge_type_picked, 
            span_m=span,
            num_spans=num_spans,
            total_length_m=total_length,
            width_m=width, 
            lanes=lanes,
            road_template=road_template,
            include_sidewalks=include_sidewalks,
            depth_of_girder=depth_of_girder,
            number_of_piers_along_length=number_of_piers_along_length,
            number_of_piers_across_width=number_of_piers_across_width,
            total_piers=total_piers,
            radius_of_pier=radius_of_pier,
            pier_type=type_of_pier,
            pier_cap_type=pier_cap_type,
            pier_cross_section=pier_cross_section,
            inner_deck_length_m=inner_deck_length,
            wing_wall_extension_m=wing_wall_extension_m,
            ))
        
    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]

