"""Compatibility entry; edit configs/other_backbones/ms1mv3_poolformer_s24_no_ln_x2_act_reverse_2block.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.other_backbones.ms1mv3_poolformer_s24_no_ln_x2_act_reverse_2block')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
