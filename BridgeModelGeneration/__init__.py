"""
Bridge Model Generation Module

Handles parametric bridge generation using CadQuery.
"""

from .bridge_pipeline import BridgePipeline
from .model_config import BridgeConfig
from .param_gen import generate_bridge_configs, configs_to_records
from .bridge_model import BridgeModel

__all__ = ['BridgePipeline', 'BridgeConfig', 'generate_bridge_configs', 'configs_to_records', 'BridgeModel']
