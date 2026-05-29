import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amc.data import RadioMLDataset
from src.amc.models import SimpleCNN1D

try:
    import pywt
except ImportError:
    pywt = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ACTIONS = [
    "no_process",
    "normalize_power",
    "wavelet_weak",
    "wavelet_strong",
    "lowpass_mild",
    "lowpass_strong",
]

SNR_GROUPS = [
    ("Hard", lambda snr: snr <= -10),
    ("Medium", lambda snr: -8 <= snr <= -2),
    ("Easy", lambda snr: snr >= 0),
]


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_limit_dataset(dataset, max_samples, seed):
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {max_samples}.")
    if max_samples >= len(dataset):
        print(f"Requested max_samples={max_samples}, using full split.")
        return dataset

    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(dataset), size=max_samples, replace=False)
    subset_indices = np.sort(subset_indices).tolist()
    print(f"Using sampled split subset: {max_samples}/{len(dataset)}, seed={seed}")
    return Subset(dataset, subset_indices)


def make_loader(dataset, batch_size, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def crop_or_pad_1d(x, length=128):
    if x.shape[0] > length:
        return x[:length]
    if x.shape[0] < length:
        return np.pad(x, (0, length - x.shape[0]), mode="constant")
    return x


def normalize_power(x):
    x = x.astype(np.float32, copy=False)
    power = np.mean(x[0] ** 2 + x[1] ** 2)
    return (x / np.sqrt(power + 1e-8)).astype(np.float32)


def wavelet_denoise_channel(channel, level, threshold_scale):
    if pywt is None:
        raise ImportError(
            "PyWavelets is required for wavelet actions. Install it with: "
            "pip install PyWavelets"
        )

    channel = channel.astype(np.float32, copy=False)
    coeffs = pywt.wavedec(channel, wavelet="db4", level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(channel.shape[0]))

    denoised_coeffs = [coeffs[0]]
    for coeff in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(coeff, threshold, mode="soft"))
    reconstructed = pywt.waverec(denoised_coeffs, wavelet="db4")
    return crop_or_pad_1d(reconstructed, length=channel.shape[0]).astype(np.float32)


def wavelet_denoise(x, level, threshold_scale):
    x = x.astype(np.float32, copy=False)
    return np.stack(
        [
            wavelet_denoise_channel(x[0], level, threshold_scale),
            wavelet_denoise_channel(x[1], level, threshold_scale),
        ],
        axis=0,
    ).astype(np.float32)


def lowpass_channel(channel, cutoff):
    channel = channel.astype(np.float32, copy=False)
    b, a = signal.butter(N=5, Wn=cutoff, btype="low")
    try:
        filtered = signal.filtfilt(b, a, channel)
    except ValueError:
        filtered = signal.lfilter(b, a, channel)
    return filtered.astype(np.float32)


def lowpass_filter(x, cutoff):
    x = x.astype(np.float32, copy=False)
    return np.stack(
        [
            lowpass_channel(x[0], cutoff),
            lowpass_channel(x[1], cutoff),
        ],
        axis=0,
    ).astype(np.float32)


def apply_action(x, action):
    if action == "no_process":
        return x.astype(np.float32, copy=False)
    if action == "normalize_power":
        return normalize_power(x)
    if action == "wavelet_weak":
        return wavelet_denoise(x, level=2, threshold_scale=0.5)
    if action == "wavelet_strong":
        return wavelet_denoise(x, level=3, threshold_scale=1.0)
    if action == "lowpass_mild":
        return lowpass_filter(x, cutoff=0.45)
    if action == "lowpass_strong":
        return lowpass_filter(x, cutoff=0.35)
    raise ValueError(f"Unknown action: {action}")


def transform_batch(x, action):
    x_np = x.cpu().numpy()
    transformed = [apply_action(sample, action) for sample in x_np]
    return torch.from_numpy(np.stack(transformed, axis=0).astype(np.float32))


def snr_group_name(snr):
    for name, predicate in SNR_GROUPS:
        if predicate(snr):
            return name
    raise ValueError(f"SNR does not belong to any configured group: {snr}")


def init_counter():
    return {"correct": 0, "total": 0}


def update_counter(counter, correct):
    counter["correct"] += int(correct)
    counter["total"] += 1


def accuracy(counter):
    total = counter["total"]
    return counter["correct"] / total if total else 0.0


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_action(model, loader, device, action, idx_to_class):
    model.eval()
    predictions = []
    correct_by_sample = {}
    overall = init_counter()
    by_snr_group = defaultdict(init_counter)
    by_modulation = defaultdict(init_counter)

    iterator = tqdm(loader, desc=action, leave=False) if tqdm else loader
    for batch in iterator:
        x = transform_batch(batch["x"], action).to(device)
        true_labels = batch["label"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top2_probs, top2_labels = torch.topk(probs, k=2, dim=1)
        pred_labels = top2_labels[:, 0]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        correct = pred_labels.eq(true_labels)

        sample_ids = batch["sample_id"].cpu().tolist()
        true_label_list = true_labels.cpu().tolist()
        pred_label_list = pred_labels.cpu().tolist()
        correct_list = correct.cpu().tolist()
        snrs = batch["snr"].cpu().tolist()
        true_modulations = list(batch["modulation"])
        top1_confidences = top2_probs[:, 0].cpu().tolist()
        top2_confidences = top2_probs[:, 1].cpu().tolist()
        entropies = entropy.cpu().tolist()

        for i, sample_id in enumerate(sample_ids):
            sample_id = int(sample_id)
            true_label = int(true_label_list[i])
            pred_label = int(pred_label_list[i])
            true_modulation = true_modulations[i]
            pred_modulation = idx_to_class[pred_label]
            snr = int(snrs[i])
            is_correct = bool(correct_list[i])
            top1_confidence = float(top1_confidences[i])
            top2_confidence = float(top2_confidences[i])

            update_counter(overall, is_correct)
            update_counter(by_snr_group[snr_group_name(snr)], is_correct)
            update_counter(by_modulation[true_modulation], is_correct)
            correct_by_sample[sample_id] = is_correct

            predictions.append(
                {
                    "action": action,
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "true_modulation": true_modulation,
                    "true_snr": snr,
                    "pred_label": pred_label,
                    "pred_modulation": pred_modulation,
                    "correct": int(is_correct),
                    "top1_confidence": f"{top1_confidence:.8f}",
                    "top2_confidence": f"{top2_confidence:.8f}",
                    "confidence_margin": f"{top1_confidence - top2_confidence:.8f}",
                    "entropy": f"{float(entropies[i]):.8f}",
                }
            )

    return {
        "predictions": predictions,
        "correct_by_sample": correct_by_sample,
        "overall": overall,
        "by_snr_group": by_snr_group,
        "by_modulation": by_modulation,
    }


def build_output_rows(action_results, actions, class_order):
    no_process_correct = action_results["no_process"]["correct_by_sample"]

    summary_rows = []
    snr_group_rows = []
    modulation_rows = []
    prediction_rows = []

    for action in actions:
        result = action_results[action]
        overall = result["overall"]
        correct_by_sample = result["correct_by_sample"]

        error_to_correct = 0
        correct_to_error = 0
        for sample_id, baseline_correct in no_process_correct.items():
            action_correct = correct_by_sample[sample_id]
            if not baseline_correct and action_correct:
                error_to_correct += 1
            elif baseline_correct and not action_correct:
                correct_to_error += 1

        summary_rows.append(
            {
                "action": action,
                "overall_accuracy": f"{accuracy(overall):.8f}",
                "correct": overall["correct"],
                "total": overall["total"],
                "error_to_correct": error_to_correct,
                "correct_to_error": correct_to_error,
                "net_gain": error_to_correct - correct_to_error,
            }
        )

        for snr_group, _ in SNR_GROUPS:
            counter = result["by_snr_group"][snr_group]
            snr_group_rows.append(
                {
                    "action": action,
                    "snr_group": snr_group,
                    "correct": counter["correct"],
                    "total": counter["total"],
                    "accuracy": f"{accuracy(counter):.8f}",
                }
            )

        for modulation in class_order:
            counter = result["by_modulation"][modulation]
            modulation_rows.append(
                {
                    "action": action,
                    "modulation": modulation,
                    "correct": counter["correct"],
                    "total": counter["total"],
                    "accuracy": f"{accuracy(counter):.8f}",
                }
            )

        prediction_rows.extend(result["predictions"])

    return summary_rows, snr_group_rows, modulation_rows, prediction_rows


def save_run_meta(path, args, actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "actions": actions,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "note": "IQ samples are transformed dynamically during inference. Processed IQ datasets are not saved.",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen fixed IQ preprocessing actions with a trained SimpleCNN1D."
    )
    parser.add_argument("--data-path", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--split-meta-path", default="data/splits/split_meta.json")
    parser.add_argument("--train-indices", default="data/splits/train_indices.npy")
    parser.add_argument("--val-indices", default="data/splits/val_indices.npy")
    parser.add_argument("--test-indices", default="data/splits/test_indices.npy")
    parser.add_argument(
        "--checkpoint-path", default="checkpoints/debug_simple_cnn_best.pt"
    )
    parser.add_argument("--output-dir", default="results/tool_screening")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=ACTIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    actions = list(dict.fromkeys(args.actions))
    if "no_process" not in actions:
        actions.insert(0, "no_process")

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Actions: {', '.join(actions)}")

    split_to_indices = {
        "train": args.train_indices,
        "val": args.val_indices,
        "test": args.test_indices,
    }
    dataset = RadioMLDataset(
        data_path=args.data_path,
        indices_path=split_to_indices[args.split],
        split_meta_path=args.split_meta_path,
        normalize=args.normalize,
    )
    dataset = maybe_limit_dataset(dataset, args.max_samples, args.seed)
    loader = make_loader(dataset, args.batch_size, args.num_workers, device)

    checkpoint = load_checkpoint(args.checkpoint_path, device)
    num_classes = int(checkpoint.get("num_classes", 11))
    model = SimpleCNN1D(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    idx_to_class_raw = checkpoint.get("idx_to_class")
    if idx_to_class_raw is None:
        idx_to_class_raw = (
            dataset.dataset.meta["idx_to_class"]
            if isinstance(dataset, Subset)
            else dataset.meta["idx_to_class"]
        )
    idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
    class_order = [idx_to_class[i] for i in range(len(idx_to_class))]

    action_results = {}
    for action in actions:
        action_results[action] = evaluate_action(
            model, loader, device, action, idx_to_class
        )
        overall = action_results[action]["overall"]
        print(
            f"{action}: accuracy={accuracy(overall):.4f} "
            f"correct={overall['correct']} total={overall['total']}"
        )

    summary_rows, snr_group_rows, modulation_rows, prediction_rows = build_output_rows(
        action_results, actions, class_order
    )

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "tool_screening_summary.csv",
        summary_rows,
        [
            "action",
            "overall_accuracy",
            "correct",
            "total",
            "error_to_correct",
            "correct_to_error",
            "net_gain",
        ],
    )
    write_csv(
        output_dir / "accuracy_by_snr_group.csv",
        snr_group_rows,
        ["action", "snr_group", "correct", "total", "accuracy"],
    )
    write_csv(
        output_dir / "accuracy_by_modulation.csv",
        modulation_rows,
        ["action", "modulation", "correct", "total", "accuracy"],
    )
    write_csv(
        output_dir / "predictions_by_action.csv",
        prediction_rows,
        [
            "action",
            "sample_id",
            "true_label",
            "true_modulation",
            "true_snr",
            "pred_label",
            "pred_modulation",
            "correct",
            "top1_confidence",
            "top2_confidence",
            "confidence_margin",
            "entropy",
        ],
    )
    save_run_meta(output_dir / "tool_screening_meta.json", args, actions)
    print(f"Saved tool screening outputs to: {output_dir}")


if __name__ == "__main__":
    main()
