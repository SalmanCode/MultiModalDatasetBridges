import random
from typing import List
from config import BridgeConfig
from dataclasses import asdict
import pandas as pd




BRIDGE_RANGES: dict[str, tuple[float, float]] = {
    "beam_slab": (20.0, 50.0),
    "box_girder": (50.0, 120.0),
}

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
        configs.append(BridgeConfig(
            bridge_id=f"bridge_{idx}", 
            bridge_type=bridge_type, 
            span_m=span, 
            width_m=width, 
            deck_m=deck, 
            lanes=lanes, 
            include_sidewalks=include_sidewalks))
        
    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]

