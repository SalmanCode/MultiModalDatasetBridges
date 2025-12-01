import random
import logging
import cadquery as cq
from dataclasses import dataclass, asdict
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



BRIDGE_RANGES: dict[str, tuple[float, float]] = {
    "beam_slab": (20.0, 50.0),
    "box_girder": (50.0, 120.0),
}


@dataclass
class BridgeConfig:
    identifier: str
    bridge_type: str
    span_m: float
    deck_m: float

def generate_bridge_configs(count: int, step: int, seed: int | None = None, overhang_m: float = 1.0) -> List[BridgeConfig]:
    rng = random.Random(seed)
    configs: List[BridgeConfig] = []
    for idx in range(1, count + 1):
        bridge_type = rng.choice(list(BRIDGE_RANGES))
        lower, upper = BRIDGE_RANGES[bridge_type]
        raw_span = rng.uniform(lower, upper)
        span = round(round(raw_span / step) * step, 2)
        deck = round(span + 2 * overhang_m, 2)
        configs.append(BridgeConfig(f"bridge_{idx}", bridge_type, span, deck))
    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]

