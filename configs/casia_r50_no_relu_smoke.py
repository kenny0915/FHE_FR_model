"""Compatibility entry; edit configs/polynomial_conversion/casia_r50_no_relu_smoke.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.polynomial_conversion.casia_r50_no_relu_smoke')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
