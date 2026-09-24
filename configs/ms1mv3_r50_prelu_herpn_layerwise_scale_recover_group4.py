"""Compatibility entry; edit configs/polynomial_conversion/ms1mv3_r50_prelu_herpn_layerwise_scale_recover_group4.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.polynomial_conversion.ms1mv3_r50_prelu_herpn_layerwise_scale_recover_group4')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
