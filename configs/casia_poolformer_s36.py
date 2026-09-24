"""Compatibility entry; edit configs/other_backbones/casia_poolformer_s36.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.other_backbones.casia_poolformer_s36')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
