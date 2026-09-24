"""Compatibility entry; edit configs/reduced_nonlinearity/ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill_lr3e4.py instead."""
from importlib import import_module as _import_module

_module = _import_module('configs.reduced_nonlinearity.ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill_lr3e4')
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
