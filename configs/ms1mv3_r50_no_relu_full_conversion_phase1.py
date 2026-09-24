"""Compatibility entry; edit configs/polynomial_conversion/ms1mv3_r50_no_relu_full_conversion_phase1.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.polynomial_conversion.ms1mv3_r50_no_relu_full_conversion_phase1')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
