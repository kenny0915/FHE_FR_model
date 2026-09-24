"""Compatibility entry; edit configs/polynomial_conversion/ms1mv3_r50_herpn_residual_scale.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.polynomial_conversion.ms1mv3_r50_herpn_residual_scale')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
