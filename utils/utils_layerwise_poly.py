"""Orchestration helpers for safe grouped polynomial calibration."""

import json
import math
import os


def activation_range_is_contained(observed_absmax, interval_radius):
    """Return true only when the observed finite maximum is inside radius."""
    observed_absmax = float(observed_absmax)
    interval_radius = float(interval_radius)
    if not math.isfinite(observed_absmax):
        return False
    if not math.isfinite(interval_radius) or interval_radius <= 0.0:
        raise ValueError("interval_radius must be finite and positive")
    return observed_absmax <= interval_radius


def fractional_group_starts_crossed(
        epoch_start, current_epoch, group_epochs, already_handled=()):
    """Return unhandled blend starts crossed inside the current epoch.

    Starts exactly at ``epoch_start`` are intentionally excluded because the
    trainer handles integer boundaries before constructing the epoch iterator.
    """
    epoch_start = float(epoch_start)
    current_epoch = float(current_epoch)
    if current_epoch < epoch_start:
        raise ValueError("current_epoch must not precede epoch_start")
    handled = set(already_handled)
    return tuple(
        group_index
        for group_index, start in enumerate(group_epochs)
        if group_index not in handled
        and epoch_start < float(start) <= current_epoch
    )


def pending_group_requires_calibration(
        uncalibrated_names, conversion_groups, completed_groups):
    """Whether the immediate next conversion group lacks an interval.

    Later groups are expected to remain uncalibrated and must not make a
    routine mid-group resume look like a newly expanded conversion frontier.
    Only the group immediately after the accepted prefix is relevant.
    """
    groups = tuple(tuple(group) for group in conversion_groups)
    completed_groups = int(completed_groups)
    if not 0 <= completed_groups <= len(groups):
        raise ValueError(
            "completed_groups must index the accepted conversion prefix")
    if completed_groups == len(groups):
        return False
    pending = set(uncalibrated_names)
    return any(name in pending for name in groups[completed_groups])


def calibrated_conversion_prefix(
        model_order, calibrated_names, conversion_groups):
    """Return the contiguous calibrated prefix inside a training frontier."""
    model_order = tuple(model_order)
    frontier = tuple(
        name for group in conversion_groups for name in tuple(group))
    if frontier != model_order[:len(frontier)]:
        raise ValueError(
            "conversion_groups must be a forward prefix of model_order")
    calibrated = set(calibrated_names)
    prefix = []
    found_gap = False
    for name in frontier:
        if name in calibrated:
            if found_gap:
                raise ValueError(
                    "calibrated conversion activations must form a prefix")
            prefix.append(name)
        else:
            found_gap = True
    return tuple(prefix)


def load_tail_replay_manifests(output, activation_names):
    """Load persisted rare-tail indices for a calibrated activation prefix.

    Hard-containment resumes must replay the same extrema that established the
    immutable intervals.  Silently continuing without a manifest would turn a
    recovery run into ordinary random sampling, so missing or malformed files
    are treated as fatal configuration/checkpoint mismatches.
    """
    results = []
    for activation_name in tuple(activation_names):
        path = os.path.join(
            os.fspath(output),
            "tail_replay_" + activation_name.replace(".", "_") + ".json",
        )
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Missing tail replay manifest for calibrated activation "
                f"{activation_name!r}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"Tail replay manifest must be an object: {path}")
        if payload.get("activation") != activation_name:
            raise ValueError(
                "Tail replay manifest activation mismatch: "
                f"expected={activation_name!r}, "
                f"observed={payload.get('activation')!r}, path={path}"
            )
        indices = payload.get("dataset_indices")
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError(
                f"Tail replay manifest has invalid dataset_indices: {path}")
        input_scale = float(payload.get("input_scale", float("nan")))
        if not math.isfinite(input_scale) or input_scale <= 0.0:
            raise ValueError(
                f"Tail replay manifest has invalid input_scale: {path}")
        results.append({
            "activation": activation_name,
            "input_scale": input_scale,
            "tail_indices": tuple(indices),
            "manifest_path": path,
        })
    return results


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
