"""Compatibility entry; edit configs/other_backbones/ms1mv3_mbf_no_relu.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.other_backbones.ms1mv3_mbf_no_relu')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
