"""Resume the interrupted four-rank NL13 PReLU-to-HerPN conversion."""

from .ms1mv3_r50_nl13_prelu_herpn import config


# The original run stopped on a TensorBoard filesystem-quota error, while the
# model and all four rank-local PartialFC checkpoints remained finite.
config.resume = True
config.tensorboard = False
