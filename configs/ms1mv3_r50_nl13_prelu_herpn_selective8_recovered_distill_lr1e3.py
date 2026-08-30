"""Higher-LR hedge for focused selective-eight embedding recovery.

This keeps the identical fully polynomial eight-site inference graph and
seven-site normalized-embedding teacher as the conservative recovery, but
uses the established base LR whose final-finetune multiplier yields an
effective ``1e-4`` backbone LR.  Strict finite guards remain enabled.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_recovered_distill import (
    config as _distill_config,
)


config = edict(_distill_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_recovered_distill_lr1e3")
config.lr = 1e-3
