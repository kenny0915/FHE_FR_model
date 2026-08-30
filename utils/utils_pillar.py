"""Training schedule helpers for PILLAR polynomial regularization."""


_PILLAR_BETA_WARMUP_DIVISORS = (100.0, 50.0, 10.0, 5.0)


def pillar_regularization_at_epoch(epoch, target_coefficient,
                                   target_exponent=10, warmup=True):
    """Return the paper's epoch-wise ``(coefficient, exponent)`` schedule.

    Epochs 0--3 use beta divided by ``100, 50, 10, 5`` and exponents
    ``4, 6, 8, 10`` (capped at the configured even target).  From epoch 4,
    both parameters are at their target values.
    """
    epoch = int(epoch)
    target_coefficient = float(target_coefficient)
    target_exponent = int(target_exponent)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if target_coefficient < 0.0:
        raise ValueError("target_coefficient must be non-negative")
    if target_exponent < 4 or target_exponent % 2:
        raise ValueError("target_exponent must be an even integer >= 4")
    if not warmup or epoch >= len(_PILLAR_BETA_WARMUP_DIVISORS):
        return target_coefficient, target_exponent
    coefficient = (
        target_coefficient / _PILLAR_BETA_WARMUP_DIVISORS[epoch])
    exponent = min(4 + 2 * epoch, target_exponent)
    return coefficient, exponent


def pillar_task_loss_weight_at_epoch(epoch, range_only_epochs=0):
    """Return the task-loss weight for PILLAR range preparation.

    The released PILLAR-ESPN ImageNet recipe uses ``classification_loss * 0``
    in epoch zero and optimizes only the activation-range penalty. Keeping the
    number of such epochs configurable preserves legacy experiment behavior.
    """
    epoch = int(epoch)
    range_only_epochs = int(range_only_epochs)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if range_only_epochs < 0:
        raise ValueError("range_only_epochs must be non-negative")
    return 0.0 if epoch < range_only_epochs else 1.0


def pillar_validation_is_strict_at_epoch(epoch, strict_start_epoch=0):
    """Return whether PILLAR validation should enforce finite/range gates."""
    epoch = int(epoch)
    strict_start_epoch = int(strict_start_epoch)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if strict_start_epoch < 0:
        raise ValueError("strict_start_epoch must be non-negative")
    return epoch >= strict_start_epoch
