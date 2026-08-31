"""Screen channel-wise tail-safe affine replacements for one PReLU."""

import argparse
import json
import sys
from pathlib import Path

import mxnet as mx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from eval import verification
from eval.compare_activation_distributions import extract_state_dict
from eval.screen_linear_replacement import MODEL_KWARGS


def load_recordio_orientations(root, record_index):
    root = Path(root)
    reader = mx.recordio.MXIndexedRecordIO(
        str(root / "train.idx"), str(root / "train.rec"), "r")
    packed = reader.read_idx(int(record_index))
    if packed is None:
        raise ValueError(f"RecordIO index {record_index} does not exist")
    _, encoded = mx.recordio.unpack(packed)
    image = mx.image.imdecode(encoded).asnumpy()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor.div(255.0).sub(0.5).div(0.5)
    return torch.stack((tensor, tensor.flip(-1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="ms1m-retinaface-t1")
    parser.add_argument("--targets", nargs="+", default=[
        "lfw", "cfp_fp", "agedb_30"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=(
            2.0, 1.0e6, 1.0e9, 1.0e12, 1.0e15, 1.0e18, 1.0e21, 1.0e23))
    parser.add_argument("--recordio-index", type=int, default=86052)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.005)
    parser.add_argument("--max-tail-growth", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-output", required=True)
    args = parser.parse_args()

    if any(threshold <= 0.0 for threshold in args.thresholds):
        raise ValueError("All thresholds must be positive")

    device = torch.device("cuda")
    model = get_model("r50_prelu_herpn", **MODEL_KWARGS)
    raw = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(raw), strict=True)
    activation = dict(model.named_modules())["layer4.2.prelu"]
    activation.set_blend(0.0)
    model.float().to(device).eval()

    channel_min = torch.full(
        (activation.weight.shape[0],), float("inf"), dtype=torch.float32)
    input_nonfinite = 0
    source_diagnostics = {"absmax": 0.0, "nonfinite": 0}

    def capture_input(_, inputs):
        nonlocal input_nonfinite
        values = inputs[0].detach().float()
        finite = torch.isfinite(values)
        input_nonfinite += int((~finite).sum().item())
        reduced = torch.where(
            finite, values, torch.full_like(values, float("inf")))
        current = reduced.amin(dim=(0, 2, 3)).cpu()
        channel_min.copy_(torch.minimum(channel_min, current))

    def capture_source_output(_, __, output):
        values = output.detach().float()
        finite = torch.isfinite(values)
        source_diagnostics["nonfinite"] += int((~finite).sum().item())
        if bool(finite.any()):
            source_diagnostics["absmax"] = max(
                source_diagnostics["absmax"],
                float(values[finite].abs().amax().item()))

    handle = activation.register_forward_pre_hook(capture_input)
    source_output_handle = model.register_forward_hook(capture_source_output)
    datasets = {}
    source = {}
    try:
        stress = load_recordio_orientations(args.data_dir, args.recordio_index)
        source_diagnostics.update(absmax=0.0, nonfinite=0)
        with torch.no_grad():
            stress_embeddings = model(stress.to(device))
        stress_finite = bool(torch.isfinite(stress_embeddings).all())
        stress_source_absmax = source_diagnostics["absmax"]
        del stress_embeddings

        for target in args.targets:
            data = verification.load_bin(
                str(Path(args.data_dir) / f"{target}.bin"), (112, 112))
            datasets[target] = data
            source_diagnostics.update(absmax=0.0, nonfinite=0)
            _, _, accuracy, std, xnorm, _ = verification.test(
                data, model, args.batch_size, 10, fail_on_nonfinite=True)
            source[target] = {
                "accuracy_flip": float(accuracy),
                "accuracy_std": float(std),
                "xnorm": float(xnorm),
                "embedding_absmax": source_diagnostics["absmax"],
                "nonfinite_elements": source_diagnostics["nonfinite"],
            }
    finally:
        handle.remove()
        source_output_handle.remove()

    teacher_slope = activation.prelu.weight.detach().reshape(-1, 1, 1)
    diagnostics = {"absmax": 0.0, "nonfinite": 0}

    def capture_output(_, __, output):
        values = output.detach().float()
        finite = torch.isfinite(values)
        diagnostics["nonfinite"] += int((~finite).sum().item())
        if bool(finite.any()):
            diagnostics["absmax"] = max(
                diagnostics["absmax"],
                float(values[finite].abs().amax().item()))

    output_handle = model.register_forward_hook(capture_output)
    candidates = []
    try:
        for threshold in args.thresholds:
            unsafe = channel_min < -float(threshold)
            with torch.no_grad():
                weights = torch.ones_like(activation.weight)
                weights[unsafe.to(device=weights.device)] = teacher_slope[
                    unsafe.to(device=teacher_slope.device)]
                activation.weight.copy_(weights)
                activation.bias.zero_()
                activation.set_blend(1.0)

            candidate = {
                "negative_tail_threshold": float(threshold),
                "teacher_slope_channels": int(unsafe.sum().item()),
                "teacher_slope_indices": unsafe.nonzero(
                    as_tuple=False).flatten().tolist(),
                "targets": {},
            }
            complete = True
            diagnostics.update(absmax=0.0, nonfinite=0)
            with torch.no_grad():
                stress_embeddings = model(stress.to(device))
            candidate["recordio"] = {
                "embedding_finite": bool(torch.isfinite(
                    stress_embeddings).all()),
                "embedding_absmax": diagnostics["absmax"],
                "nonfinite_elements": diagnostics["nonfinite"],
            }
            complete = complete and candidate["recordio"]["embedding_finite"]
            del stress_embeddings
            for target, data in datasets.items():
                diagnostics.update(absmax=0.0, nonfinite=0)
                try:
                    _, _, accuracy, std, xnorm, _ = verification.test(
                        data, model, args.batch_size, 10,
                        fail_on_nonfinite=True)
                    metrics = {
                        "accuracy_flip": float(accuracy),
                        "accuracy_std": float(std),
                        "xnorm": float(xnorm),
                        "embedding_absmax": diagnostics["absmax"],
                        "nonfinite_elements": diagnostics["nonfinite"],
                    }
                except FloatingPointError as error:
                    complete = False
                    metrics = {
                        "error": str(error),
                        "embedding_absmax": diagnostics["absmax"],
                        "nonfinite_elements": diagnostics["nonfinite"],
                    }
                candidate["targets"][target] = metrics
            candidate["complete"] = complete
            candidates.append(candidate)
    finally:
        output_handle.remove()

    accepted = []
    for candidate in candidates:
        if not candidate["complete"]:
            continue
        drops = [
            source[target]["accuracy_flip"]
            - candidate["targets"][target]["accuracy_flip"]
            for target in args.targets
        ]
        candidate["accuracy_drops"] = dict(zip(args.targets, drops))
        target_tail_safe = all(
            candidate["targets"][target]["embedding_absmax"]
            <= source[target]["embedding_absmax"] * args.max_tail_growth
            for target in args.targets
        )
        stress_tail_safe = (
            candidate["recordio"]["embedding_absmax"]
            <= stress_source_absmax * args.max_tail_growth
        )
        candidate["tail_growth_gate"] = {
            "targets": target_tail_safe,
            "recordio": stress_tail_safe,
        }
        if (max(drops) <= args.max_accuracy_drop
                and target_tail_safe and stress_tail_safe):
            accepted.append(candidate)
    if not accepted:
        result = {
            "source": source,
            "calibration": {
                "recordio_index": args.recordio_index,
                "recordio_embedding_finite": stress_finite,
                "recordio_embedding_absmax": stress_source_absmax,
                "input_nonfinite_elements": input_nonfinite,
                "channel_min": channel_min.tolist(),
            },
            "candidates": candidates,
            "selected_threshold": None,
            "checkpoint_output": None,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"No channel-wise candidate passed the accuracy/tail gates; "
            f"diagnostics saved to {output}")
    selected = max(
        accepted,
        key=lambda item: sum(
            item["targets"][target]["accuracy_flip"]
            for target in args.targets),
    )

    selected_threshold = selected["negative_tail_threshold"]
    unsafe = channel_min < -selected_threshold
    with torch.no_grad():
        weights = torch.ones_like(activation.weight)
        device_mask = unsafe.to(device=weights.device)
        weights[device_mask] = teacher_slope[device_mask]
        activation.weight.copy_(weights)
        activation.bias.zero_()
        activation.set_blend(1.0)
    checkpoint_output = Path(args.checkpoint_output)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint_output,
    )

    result = {
        "source": source,
        "calibration": {
            "recordio_index": args.recordio_index,
            "recordio_embedding_finite": stress_finite,
            "recordio_embedding_absmax": stress_source_absmax,
            "input_nonfinite_elements": input_nonfinite,
            "channel_min": channel_min.tolist(),
        },
        "candidates": candidates,
        "selected_threshold": selected_threshold,
        "checkpoint_output": str(checkpoint_output),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
