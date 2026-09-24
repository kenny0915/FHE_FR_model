"""Compatibility entry; edit configs/polynomial_conversion/ms1mv3_r50_pillar_espn_scale4_early.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.polynomial_conversion.ms1mv3_r50_pillar_espn_scale4_early')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
