"""Resume after group 2 needs a second range-conditioning epoch.

The failed strict pass occurs after the epoch-4 full checkpoints have already
been saved. Moving group 2's blend start from epoch 4 to epoch 5 makes epoch 4
another zero-blend conditioning epoch. Every later group moves by one epoch,
while ``num_epoch`` remains 22; this trades one final joint-tuning epoch for
conditioning and does not extend the total training schedule.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_group4_resume_epoch3 import (
    config as resume_epoch3_config,
)


config = edict(resume_epoch3_config.copy())
config.herpn_group_epochs = (2, 5, 7, 9, 11, 13, 15)
