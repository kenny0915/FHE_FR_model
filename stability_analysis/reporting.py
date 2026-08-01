"""JSON, CSV, and human-readable reporting for stability studies."""

import csv
import json
from pathlib import Path


def _number(value, digits=4):
    if value is None:
        return "n/a"
    return ("{:.%dg}" % digits).format(value)


def _percent(value):
    return "n/a" if value is None else "{:.2f}%".format(100.0 * value)


def _layer_csv_row(layer):
    behavior = layer["model_behavior"]
    derivative = layer["local_derivative"]
    shift = layer["downstream_distribution_shift"]
    input_distribution = layer["input_distribution"]
    return {
        "index": layer["index"],
        "name": layer["name"],
        "stage": layer["stage"],
        "teacher": layer["teacher"],
        "outside_interval_fraction": input_distribution[
            "outside_interval_fraction"],
        "input_std": input_distribution["std"],
        "input_absmax": input_distribution["absmax"],
        "local_relative_rmse": layer["local_approximation"]["relative_rmse"],
        "derivative_negative_fraction": derivative.get("negative_fraction"),
        "derivative_absmax": derivative.get("absmax"),
        "embedding_cosine_mean": behavior["embedding_cosine_mean"],
        "embedding_relative_rmse": behavior["embedding_relative_rmse"],
        "pairwise_cosine_mae": behavior["pairwise_cosine_mae"],
        "embedding_norm_ratio_mean": behavior["embedding_norm_ratio_mean"],
        "downstream_probe": layer["downstream_probe"],
        "downstream_std_ratio": shift["std_ratio"],
        "downstream_absmax_ratio": shift["absmax_ratio"],
    }


def render_markdown(report):
    run = report.get("run", {})
    dataset_kind = run.get("dataset_kind", run.get("dataset", "unknown"))
    synthetic = dataset_kind == "synthetic"
    lines = [
        "# Activation stability study",
        "",
        "This report isolates one activation replacement at a time. "
        "It uses the trained checkpoint, not random model weights.",
        "",
    ]
    if synthetic:
        lines.extend([
            "> **Smoke-result warning:** the local checkout has no face images. "
            "These measurements use deterministic normalized synthetic inputs. "
            "Use them to validate wiring and identify mechanisms, not to choose "
            "a production interval or claim recognition accuracy.",
            "",
        ])
    lines.extend([
        "## Run",
        "",
        "- Model: `{}`".format(run.get("model", "unknown")),
        "- Checkpoint: `{}`".format(run.get("checkpoint", "unknown")),
        "- Dataset: `{}`".format(run.get("dataset", "unknown")),
        "- Samples: {} in {} batch(es)".format(
            report["samples_analyzed"], report["batches_analyzed"]),
        "- Activations: {}".format(report["activation_count"]),
        "- Replacement family: `{}`".format(
            run.get("replacement", "custom/default HerPN")),
        "- Monitored interval: `{}` (monitoring only; no FHE-unfriendly clamp)".format(
            report["monitored_interval"]),
        "",
        "## Per-layer replacement data",
        "",
        "| # | activation | outside | input std | input absmax | local rel. RMSE | negative poly derivative | embedding cosine | pairwise cosine MAE | next std ratio |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for layer in report["layer_results"]:
        behavior = layer["model_behavior"]
        derivative = layer["local_derivative"]
        shift = layer["downstream_distribution_shift"]
        lines.append(
            "| {index} | `{name}` | {outside} | {std} | {absmax} | {local} | "
            "{negative} | {cosine} | {pairwise} | {std_ratio} |".format(
                index=layer["index"],
                name=layer["name"],
                outside=_percent(layer["input_distribution"][
                    "outside_interval_fraction"]),
                std=_number(layer["input_distribution"]["std"]),
                absmax=_number(layer["input_distribution"]["absmax"]),
                local=_number(layer["local_approximation"]["relative_rmse"]),
                negative=_percent(derivative.get("negative_fraction")),
                cosine=_number(behavior["embedding_cosine_mean"], 6),
                pairwise=_number(behavior["pairwise_cosine_mae"]),
                std_ratio=_number(shift["std_ratio"]),
            ))

    smallest_std = min(
        report["layer_results"],
        key=lambda layer: layer["input_distribution"]["std"])
    largest_local_error = max(
        report["layer_results"],
        key=lambda layer: layer["local_approximation"]["relative_rmse"])
    largest_shift = max(
        report["layer_results"],
        key=lambda layer: layer["downstream_distribution_shift"]["std_ratio"])

    lines.extend([
        "",
        "`outside` is the fraction of baseline activation inputs outside the "
        "monitored interval. `next std ratio` compares the next activation's "
        "input standard deviation after versus before this one replacement.",
        "",
        "The smallest sampled input standard deviation is `{}` at `{}`; its "
        "replacement constant-term absolute mean is `{}`. The largest local "
        "relative error is `{}` at `{}`, and the largest next-layer standard-"
        "deviation multiplier is `{}` from `{}`.".format(
            _number(smallest_std["input_distribution"]["std"]),
            smallest_std["name"],
            _number(smallest_std["replacement"]["constant_absmean"]),
            _number(largest_local_error["local_approximation"][
                "relative_rmse"]),
            largest_local_error["name"],
            _number(largest_shift["downstream_distribution_shift"][
                "std_ratio"]),
            largest_shift["name"],
        ),
        "",
        "## Approximation interval sweep",
        "",
        "This sweep refits a uniform-L2 quadratic for every interval. It is "
        "separate from HerPN, whose Gaussian weighting does not change when its "
        "monitoring interval changes.",
        "",
        "| interval | mean outside | inside RMSE | outside RMSE | worst outside error | mean observed rel. RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["interval_summary"]:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row["interval"],
                _percent(row["mean_outside_fraction"]),
                _number(row["mean_inside_rmse"]),
                _number(row["mean_outside_rmse"]),
                _number(row["worst_outside_absolute_error"]),
                _number(row["mean_observed_relative_rmse"]),
            ))

    hints = report["data_driven_hints"]
    least = hints["first_replacement_candidates"]
    defer = hints["defer_or_use_longer_transition"]
    lines.extend([
        "",
        "The lowest sampled mean error in this interval family is `{}`. This "
        "is calibration evidence only, especially for synthetic data.".format(
            hints["lowest_mean_error_interval_in_this_sweep"]),
    ])

    all_replaced = report["all_replaced"]
    all_behavior = all_replaced["model_behavior"]
    lines.extend([
        "",
        "## All activations replaced",
        "",
        "- Embedding cosine mean: `{}`".format(
            _number(all_behavior["embedding_cosine_mean"], 6)),
        "- Embedding relative RMSE: `{}`".format(
            _number(all_behavior["embedding_relative_rmse"])),
        "- Pairwise cosine MAE: `{}`".format(
            _number(all_behavior["pairwise_cosine_mae"])),
        "- Non-finite embedding fraction: `{}`".format(
            _percent(all_behavior["nonfinite_fraction"])),
        "- Layers receiving non-finite inputs: `{}`".format(
            all_replaced["layers_with_nonfinite_inputs"]),
        "- First interval violation: `{}`".format(
            all_replaced["first_interval_violation"]),
        "- First input absmax above 100: `{}`".format(
            all_replaced["first_absmax_over_100"]),
        "- First non-finite activation input: `{}`".format(
            all_replaced["first_nonfinite_input"]),
        "",
        "This cumulative probe uses no retraining. Compare it with the isolated "
        "rows to see how repeated distribution shifts compound.",
        "",
    ])

    lines.extend([
        "## Design and training insights",
        "",
        "1. **Treat interval selection as a tail-risk problem.** A narrow fit "
        "reduces central error but increases the quadratic coefficient, so rare "
        "outliers create large outputs and derivatives. A wider fit lowers tail "
        "growth but spends approximation capacity on values the layer may rarely "
        "see. Select intervals from real per-layer quantiles, then check held-out "
        "tail errors rather than only uniform-grid error.",
        "2. **Do not interpret HerPN's `[-R, R]` label as a clamp or minimax "
        "guarantee.** This HerPN is Gaussian-weighted. Changing only `R` changes "
        "the warning threshold, not its coefficients. Refit or rescale the "
        "polynomial if the interval is meant to affect approximation behavior.",
        "3. **Polynomial instability comes from value and derivative "
        "amplification.** Degree-2 tails grow as `x^2`; their derivative grows "
        "linearly and can reverse sign on negative inputs. Residual blocks then "
        "carry this shift into later BatchNorm statistics, and repeated "
        "replacements compound a small local error into embedding drift.",
        "4. **Use progressive conversion.** On this calibration run, begin with "
        "`{}`; defer `{}` or give them longer transitions. Re-rank with real "
        "faces because this order is data-dependent.".format(
            "`, `".join(least), "`, `".join(defer)),
        "5. **Train each student locally before relying on task loss.** Initialize "
        "coefficients by regression on that layer's recorded inputs, retain a "
        "teacher-activation distillation loss, convert small groups, refresh "
        "normalization statistics, lower the backbone learning rate, clip "
        "gradients, and retain interval/tail penalties during training. Such "
        "penalties may use non-polynomial operations because they are outside "
        "encrypted inference.",
        "6. **Validate behavior, not only activation MSE.** After every group, "
        "check embedding cosine, pairwise-score drift, LFW/CFP-FP/AgeDB accuracy, "
        "non-finite gradients, activation quantiles, and the final folded "
        "polynomial graph.",
        "",
        "## Recommended GPU-server sequence",
        "",
        "1. Run this study on 5k-20k held-out normalized faces.",
        "2. Choose per-stage or per-layer intervals from train calibration data; "
        "verify tails on a disjoint set.",
        "3. Fit all students locally with the baseline frozen.",
        "4. Convert the lowest-impact groups progressively with distillation.",
        "5. Recalibrate BatchNorm after each group and evaluate verification sets.",
        "6. Fully fold the polynomial model and repeat range/accuracy checks.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report, json_path, markdown_path=None, csv_path=None):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report) + "\n")

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [_layer_csv_row(layer) for layer in report["layer_results"]]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
