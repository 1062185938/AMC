import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amc.data import RadioMLDataset
from src.amc.models import SimpleCNN1D

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_loader(dataset, batch_size, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def init_counter():
    return {"correct": 0, "total": 0}


def update_counter(counter, correct):
    counter["total"] += 1
    counter["correct"] += int(correct)


def counter_to_rows(counter_by_key, key_name):
    rows = []
    for key in sorted(counter_by_key):
        counter = counter_by_key[key]
        total = counter["total"]
        correct = counter["correct"]
        rows.append(
            {
                key_name: key,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def evaluate_split(model, dataset, loader, device, output_dir, split_name):
    model.eval()
    idx_to_class = {int(k): v for k, v in dataset.meta["idx_to_class"].items()}

    total = 0
    correct_total = 0
    by_snr = defaultdict(init_counter)
    by_modulation = defaultdict(init_counter)
    prediction_rows = []

    iterator = tqdm(loader, desc=f"eval-{split_name}", leave=False) if tqdm else loader
    for batch in iterator:
        x = batch["x"].to(device)
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
            true_label = int(true_label_list[i])
            pred_label = int(pred_label_list[i])
            true_modulation = true_modulations[i]
            pred_modulation = idx_to_class[pred_label]
            snr = int(snrs[i])
            is_correct = bool(correct_list[i])

            total += 1
            correct_total += int(is_correct)
            update_counter(by_snr[snr], is_correct)
            update_counter(by_modulation[true_modulation], is_correct)

            top1_confidence = float(top1_confidences[i])
            top2_confidence = float(top2_confidences[i])
            prediction_rows.append(
                {
                    "sample_id": int(sample_id),
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

    overall_accuracy = correct_total / total if total else 0.0
    metrics = {
        "split": split_name,
        "overall_accuracy": overall_accuracy,
        "correct": correct_total,
        "total": total,
    }

    prediction_fields = [
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
    ]
    metric_fields_snr = ["snr", "correct", "total", "accuracy"]
    metric_fields_mod = ["modulation", "correct", "total", "accuracy"]

    prediction_path = output_dir / f"{split_name}_predictions.csv"
    overall_path = output_dir / f"{split_name}_metrics_overall.json"
    by_snr_path = output_dir / f"{split_name}_accuracy_by_snr.csv"
    by_mod_path = output_dir / f"{split_name}_accuracy_by_modulation.csv"
    snr_curve_path = output_dir / f"{split_name}_snr_accuracy_curve.csv"

    by_snr_rows = counter_to_rows(by_snr, "snr")
    by_mod_rows = counter_to_rows(by_modulation, "modulation")

    write_csv(prediction_path, prediction_rows, prediction_fields)
    write_json(overall_path, metrics)
    write_csv(by_snr_path, by_snr_rows, metric_fields_snr)
    write_csv(snr_curve_path, by_snr_rows, metric_fields_snr)
    write_csv(by_mod_path, by_mod_rows, metric_fields_mod)

    print(
        f"{split_name}: accuracy={overall_accuracy:.4f} "
        f"correct={correct_total} total={total}"
    )
    print(f"{split_name}: predictions saved to {prediction_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SimpleCNN1D baseline.")
    parser.add_argument("--data-path", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--split-meta-path", default="data/splits/split_meta.json")
    parser.add_argument("--checkpoint-path", default="checkpoints/simple_cnn_best.pt")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--train-indices", default="data/splits/train_indices.npy")
    parser.add_argument("--val-indices", default="data/splits/val_indices.npy")
    parser.add_argument("--test-indices", default="data/splits/test_indices.npy")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.checkpoint_path, device)
    model = SimpleCNN1D(num_classes=int(checkpoint.get("num_classes", 11)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    split_to_indices = {
        "train": args.train_indices,
        "val": args.val_indices,
        "test": args.test_indices,
    }
    output_dir = Path(args.results_dir)

    all_metrics = {}
    for split_name in args.splits:
        dataset = RadioMLDataset(
            data_path=args.data_path,
            indices_path=split_to_indices[split_name],
            split_meta_path=args.split_meta_path,
            normalize=args.normalize,
        )
        loader = make_loader(dataset, args.batch_size, args.num_workers, device)
        all_metrics[split_name] = evaluate_split(
            model, dataset, loader, device, output_dir, split_name
        )

    write_json(output_dir / "metrics_summary.json", all_metrics)
    print(f"Metrics summary saved to: {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
