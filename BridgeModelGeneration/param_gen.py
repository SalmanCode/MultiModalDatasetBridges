import random
import math
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
    # rural and footway sections for narrow bridges
    "footway":     {"lanes": 1, "lane_w": 2.00, "shoulder": 0.0, "edge": 0.15, "cap_range": (0.15, 0.50)},
    "rural_1lane": {"lanes": 1, "lane_w": 3.25, "shoulder": 0.0, "edge": 0.25, "cap_range": (0.25, 0.75)},
    "rural_2lane": {"lanes": 2, "lane_w": 2.75, "shoulder": 0.0, "edge": 0.25, "cap_range": (0.25, 0.75)},
}

# currently we are not using the road template weights: all are equally sampled.

# Bridge families fitted to 32 measured real bridges.
#   lw   = (geometric mean, log-sd) of L/W, fitted per family
#   abut = (total_length - inner_deck_length) / total_length, fitted per family
#   clr  = bridge clearance height range
FAMILIES = {
    # saeed: broad, squat river crossings. Two thirds of the length is abutment.
    "wide_crossing":  dict(w=0.20, templates="dual",
                           lw=(0.70, 0.24), spans=((1,), (1.0,)),
                           abut=(0.62, 0.76), clr=(3.5, 6.0)),
    # japan: narrow rural crossings and footbridges
    "minor_road":     dict(w=0.22, templates=("footway", "rural_1lane", "rural_2lane", "RQ9"),
                           lw=(2.67, 0.57), spans=((1,), (1.0,)),
                           abut=(0.18, 0.60), clr=(1.0, 4.0)),
    # bamberg: standard deck, one span, heavy wing walls
    "single_span_rq": dict(w=0.20, templates=("RQ9", "RQ11", "RQ11_5", "RQ15_5", "RQ31_one_deck"),
                           lw=(2.13, 0.45), spans=((1,), (1.0,)),
                           abut=(0.42, 0.80), clr=(3.0, 5.5)),
    # cambridge: multi-span viaducts, plain abutments
    "multi_span_rq":  dict(w=0.38, templates=("RQ9", "RQ11", "RQ11_5", "RQ15_5",
                                              "RQ31_one_deck", "RQ36_one_deck"),
                           lw=(4.17, 0.46), spans=((2, 3, 4, 5), (0.15, 0.25, 0.45, 0.15)),
                           abut=(0.03, 0.25), clr=(4.5, 6.5)),
}

# two carriageways with a median, for wide crossings
DUAL_TEMPLATES = ("RQ31_one_deck", "RQ36_one_deck")
DUAL_MEDIAN_RANGE = (3.0, 14.0)

# bounds from real bridges, out of range draws are resampled
LENGTH_RANGE = (5.0, 75.0)
LW_RANGE     = (0.45, 7.50)
SPAN_RANGE   = (4.0, 35.0)

# end span / interior span ratio from real bridges
END_SPAN_RATIO = (0.46, 0.83)

# Column cross-sections, from the real bank's column_shape annotation.
COLUMN_SHAPES = ("cylindrical", "rectangular", "trapezoidal")

# Spill-through slope, horizontal : vertical (real ~1.55 - 2.5).
SPILL_SLOPE_RANGE = (1.5, 2.5)

# Single span: span_m is bearing-to-bearing, inner_deck_length_m face-to-face.
SINGLE_SPAN_OVER_OPENING = (1.05, 1.30)

# minimum deck depth
DEPTH_FLOOR = (0.35, 0.80)
DEPTH_RANGE = (0.22, 2.90)

# slab and girder (T-beam) decks
BRIDGE_SPECS = {
    "slab": {
        # slabs are shallower and shorter (max ~20 m span)
        "depth_ratio": (0.045, 0.075),
        "max_span": 20.0,
    },
    "girder": {
        # Real depth/span is 0.048 - 0.117 (median 0.079).
        "depth_ratio": (0.065, 0.115),
        "max_span": SPAN_RANGE[1],
    },
}


def pick_span(bridge_type: str, rng: random.Random, family: str, width_m: float):
    """Sample the length from L/W and width, then derive the spans. Returns None if not buildable."""
    spec = FAMILIES[family]
    lw = math.exp(rng.normalvariate(math.log(spec["lw"][0]), spec["lw"][1]))
    if not (LW_RANGE[0] <= lw <= LW_RANGE[1]):
        return None
    total = lw * width_m
    if not (LENGTH_RANGE[0] <= total <= LENGTH_RANGE[1]):
        return None

    num_spans = rng.choices(spec["spans"][0], weights=spec["spans"][1], k=1)[0]

    # abutment share of the total length
    frac = rng.uniform(*spec["abut"])
    inner = total * (1.0 - frac)
    ext = (total - inner) / 2.0     # wing wall / abutment allowance per side
    if inner <= 1.0 or ext <= 0.25:
        return None

    if num_spans == 1:
        span = inner * rng.uniform(*SINGLE_SPAN_OVER_OPENING)
        if not (SPAN_RANGE[0] <= span <= SPAN_RANGE[1]):
            return None
        layout = [inner]
    elif num_spans == 2:
        # two spans: one pier, no interior span
        r = rng.uniform(0.75, 1.0)
        a = inner * r / (1.0 + r)
        b = inner - a
        if not (SPAN_RANGE[0] <= min(a, b) and max(a, b) <= SPAN_RANGE[1]):
            return None
        span, layout = max(a, b), [a, b]
    else:
        # a, s, s, ..., s, a with a = rho * s -> end spans shorter than interior.
        rho = rng.uniform(*END_SPAN_RATIO)
        interior = inner / (2 * rho + num_spans - 2)
        end = rho * interior
        if not (SPAN_RANGE[0] <= interior <= SPAN_RANGE[1]) or end < 3.0:
            return None
        span, layout = interior, [end] + [interior] * (num_spans - 2) + [end]

    if max(layout) > BRIDGE_SPECS[bridge_type]["max_span"]:
        return None

    # depth from the modelled span, not the bearing span
    ratio = rng.uniform(*BRIDGE_SPECS[bridge_type]["depth_ratio"])
    depth = max(max(layout) * ratio, rng.uniform(*DEPTH_FLOOR))
    depth = min(max(depth, DEPTH_RANGE[0]), DEPTH_RANGE[1])

    return (round(span, 2), num_spans, round(inner, 2), round(total, 2),
            round(depth, 2), round(ext, 2), [round(v, 2) for v in layout])


def pick_deck_width(rng: random.Random, family: str):
    """Sample width from the German ROAD_TEMPLATES entries this family uses."""
    spec = FAMILIES[family]

    if spec["templates"] == "dual":
        name = rng.choice(DUAL_TEMPLATES)
        t = ROAD_TEMPLATES[name]
        roadway_width = t["lanes"] * t["lane_w"] + 2 * (t["shoulder"] + t["edge"])
        cap = rng.uniform(*t["cap_range"])
        width = 2 * roadway_width + rng.uniform(*DUAL_MEDIAN_RANGE) + 2 * cap
        return round(width, 2), 2 * t["lanes"], name + "_dual", cap > 1.0

    name = rng.choice(spec["templates"])
    t = ROAD_TEMPLATES[name]
    cap = rng.uniform(*t["cap_range"])
    roadway_width = t["lanes"] * t["lane_w"] + 2 * (t["shoulder"] + t["edge"])
    width = round(roadway_width + 2 * cap, 2)

    include_sidewalks = cap > 1.0

    return width, t["lanes"], name, include_sidewalks



def piers_combination(lanes: int, rng: random.Random, bridge_type: str, width_m: float, depth_of_girder: float, num_spans: int) -> int:
    """Pier layout: multicolumn or wall."""
    num_of_piers_along_length = num_spans - 1
    if num_of_piers_along_length == 0:
        return 0, 0, 0.6, None, None, None

    type_of_pier = rng.choices(["multicolumn", "wall"], weights=[0.6, 0.4], k=1)[0]
    radius_of_pier = round(rng.uniform(0.40, 0.70), 2)   # real columns 0.80 - 1.40 m dia.

    if type_of_pier == "wall":
        # A wall pier carries the deck directly: no cap, no columns.
        return num_of_piers_along_length, 1, radius_of_pier, "wall", None, None

    # The three column shapes are equally common in the real bank.
    pier_cross_section = rng.choice(COLUMN_SHAPES)
    if width_m >= 20.0:
        num_of_piers_across_width = int(min(max(round(width_m / 5.0), 3), 6))
    else:
        num_of_piers_across_width = rng.choices((2, 3), weights=(0.65, 0.35), k=1)[0]

    # Trapezoidal columns are blades that flare into the slab: no cap.
    cap = None if pier_cross_section == "trapezoidal" else "prismatic"
    return (num_of_piers_along_length, num_of_piers_across_width, radius_of_pier,
            "multicolumn", cap, pier_cross_section)


def generate_bridge_configs(count: int, bridge_type: str, seed: int | None = None) -> List[BridgeConfig]:
    rng = random.Random(seed)
    configs: List[BridgeConfig] = []
    families = list(FAMILIES)
    family_weights = [FAMILIES[f]["w"] for f in families]

    for idx in range(1, count + 1):
        for _ in range(200):        # rejection sampling: resample, never clip
            family = rng.choices(families, weights=family_weights, k=1)[0]
            bridge_type_picked = bridge_type or rng.choice(list(BRIDGE_SPECS.keys()))
            width, lanes, road_template, include_sidewalks = pick_deck_width(rng, family)

            picked = pick_span(bridge_type_picked, rng, family, width)
            if picked is None:
                continue
            span, num_spans, inner_deck_length, total_length, depth_of_girder, ext, layout = picked

            clearance = round(rng.uniform(*FAMILIES[family]["clr"]), 2)

            (number_of_piers_along_length, number_of_piers_across_width, radius_of_pier,
             type_of_pier, pier_cap_type, pier_cross_section) = piers_combination(
                lanes, rng, bridge_type_picked, width, depth_of_girder, num_spans)
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
                wing_wall_extension_m=ext,
                bridge_clearance_height=clearance,
                family=family,
                span_layout_m=layout,
                abutment_slope=round(rng.uniform(*SPILL_SLOPE_RANGE), 2),
                ))
            break
        else:
            raise RuntimeError(f"could not sample a buildable bridge {idx} in 200 tries")

    return configs

def save_bridge_configs(configs: List[BridgeConfig], file_path: str) -> None:
    df = pd.DataFrame(asdict(config) for config in configs)
    df.to_excel(file_path, index=False) 

def configs_to_records(configs):
    return [asdict(cfg) for cfg in configs]
