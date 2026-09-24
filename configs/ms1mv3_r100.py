"""Compatibility entry; edit configs/baseline/ms1mv3_r100.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.baseline.ms1mv3_r100')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
