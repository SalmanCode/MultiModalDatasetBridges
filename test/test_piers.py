
import logging
from geometry.bridge_model import BridgeModel
from config import BridgeConfig
import math
import pytest

def test_compute_pier_positions():
    cfg = BridgeConfig(
        bridge_id="bridge_test",
        bridge_type="beam_slab",
        span_m=30.0,
        width_m=12.0,
        num_spans = 5,
        total_length_m=150.0,
        lanes=3,
        depth_of_girder=2.0,
        include_sidewalks = False
    )
    
    model = BridgeModel(cfg)
    positions = model.compute_pier_positions()
    
    spans = []
    
    spans.append(positions[0] - cfg.total_length_m / 2) # first span
    spans.append(positions[1] - positions[0]) # second span
    first_span = spans[0]
    second_span = spans[1]
    print(f"first span: {first_span}, second span: {second_span}")
    
    assert math.isclose(spans[0] / spans[1], 0.7, rel_tol=1e-2)