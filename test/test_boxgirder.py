import logging
from geometry.bridge_model import BridgeModel
from config import BridgeConfig
import math
import pytest

def test_compute_box_girder_spacing():
    cfg = BridgeConfig(
        bridge_id="bridge_test",
        bridge_type="box_girder",
        span_m=30.0,
        width_m=12.0,
        num_spans = 5,
        total_length_m=150.0,
        lanes=3,
        depth_of_girder=2.0,
        include_sidewalks = False,
        number_of_piers_per_lane=2,
        radius_of_pier=1.0,
        pier_type="circular",
        pier_cap_type="circular",
        pier_cross_section="circular",
    )
    
    model = BridgeModel(cfg)
    num_of_cells = model.compute_box_girder_spacing(cfg.width_m)
    assert num_of_cells == 2

    cfg.depth_of_girder = 4.0
    model = BridgeModel(cfg)
    num_of_cells = model.compute_box_girder_spacing(cfg.width_m)
    assert num_of_cells == 1

    cfg.depth_of_girder = 1.0
    model = BridgeModel(cfg)
    num_of_cells = model.compute_box_girder_spacing(cfg.width_m)
    assert num_of_cells == 3