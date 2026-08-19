"""Orchestration helpers for safe grouped polynomial calibration."""


def causally_calibrate_polynomial_group(
        module, activation_names, calibrate_one, verify_group):
    """Calibrate a forward-ordered group on its actual polynomial prefix.

    ``calibrate_one`` profiles one still-PReLU activation.  Each calibrated
    activation is then enabled temporarily, so the next activation observes
    the same polynomial prefix it will receive during conversion.  Original
    blends are restored even when calibration or verification fails.
    """
    names = tuple(activation_names)
    if not names:
        raise ValueError("Causal polynomial calibration group is empty")

    activations = dict(module.named_modules())
    missing = [name for name in names if name not in activations]
    if missing:
        raise ValueError(
            f"Unknown causal polynomial calibration activations: {missing}")
    original_blends = {
        name: float(activations[name].blend.item()) for name in names
    }
    if any(abs(blend) > 1e-9 for blend in original_blends.values()):
        raise RuntimeError(
            "Causal strict calibration must run at the zero-blend boundary")

    results = []
    try:
        for index, name in enumerate(names):
            results.extend(calibrate_one(name, index, len(names)))
            activations[name].set_blend(1.0)
        verification = verify_group(names)
    finally:
        for name, blend in original_blends.items():
            activations[name].set_blend(blend)
    return results, verification
