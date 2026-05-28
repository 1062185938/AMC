import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SNR_GROUPS = {
    "hard": lambda snr: snr <= -10,
    "medium": lambda snr: -8 <= snr <= -2,
    "easy": lambda snr: snr >= 0,
}


def read_predictions(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "true_modulation",
            "true_snr",
            "pred_modulation",
            "correct",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {path}: {sorted(missing_fields)}"
            )
        return list(reader)


def load_class_order(split_meta_path, prediction_rows):
    split_meta_path = Path(split_meta_path)
    if split_meta_path.exists():
        with split_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        idx_to_class = meta.get("idx_to_class", {})
        if idx_to_class:
            return [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    classes = set()
    for row in prediction_rows:
        classes.add(row["true_modulation"])
        classes.add(row["pred_modulation"])
    return sorted(classes)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_accuracy_by_modulation_snr(rows, class_order):
    counters = defaultdict(lambda: {"correct": 0, "total": 0})
    snrs = set()
    for row in rows:
        modulation = row["true_modulation"]
        snr = int(row["true_snr"])
        correct = int(row["correct"])
        key = (modulation, snr)
        counters[key]["correct"] += correct
        counters[key]["total"] += 1
        snrs.add(snr)

    ordered_snrs = sorted(snrs)
    output_rows = []
    for modulation in class_order:
        for snr in ordered_snrs:
            counter = counters.get((modulation, snr), {"correct": 0, "total": 0})
            total = counter["total"]
            correct = counter["correct"]
            output_rows.append(
                {
                    "modulation": modulation,
                    "snr": snr,
                    "correct": correct,
                    "total": total,
                    "accuracy": f"{(correct / total if total else 0.0):.8f}",
                }
            )
    return output_rows


def init_matrix(class_order):
    return {
        true_mod: {pred_mod: 0 for pred_mod in class_order}
        for true_mod in class_order
    }


def build_confusion_matrix(rows, class_order):
    matrix = init_matrix(class_order)
    for row in rows:
        true_mod = row["true_modulation"]
        pred_mod = row["pred_modulation"]
        matrix[true_mod][pred_mod] += 1
    return matrix


def write_matrix(path, matrix, class_order, normalized=False):
    fieldnames = ["true_modulation"] + class_order
    output_rows = []
    for true_mod in class_order:
        counts = matrix[true_mod]
        total = sum(counts.values())
        row = {"true_modulation": true_mod}
        for pred_mod in class_order:
            value = counts[pred_mod]
            if normalized:
                row[pred_mod] = f"{(value / total if total else 0.0):.8f}"
            else:
                row[pred_mod] = value
        output_rows.append(row)
    write_csv(path, output_rows, fieldnames)


def snr_group_name(snr):
    for name, predicate in SNR_GROUPS.items():
        if predicate(snr):
            return name
    raise ValueError(f"SNR does not belong to any configured group: {snr}")


def split_rows_by_snr_group(rows):
    grouped = {name: [] for name in SNR_GROUPS}
    for row in rows:
        group_name = snr_group_name(int(row["true_snr"]))
        grouped[group_name].append(row)
    return grouped


def build_top3_by_snr_group(grouped_rows, class_order):
    output_rows = []
    for group_name in SNR_GROUPS:
        matrix = build_confusion_matrix(grouped_rows[group_name], class_order)
        for true_mod in class_order:
            counts = matrix[true_mod]
            total = sum(counts.values())
            mistakes = [
                (pred_mod, count)
                for pred_mod, count in counts.items()
                if pred_mod != true_mod and count > 0
            ]
            mistakes.sort(key=lambda item: (-item[1], class_order.index(item[0])))
            for rank, (pred_mod, count) in enumerate(mistakes[:3], start=1):
                output_rows.append(
                    {
                        "snr_group": group_name,
                        "true_modulation": true_mod,
                        "rank": rank,
                        "pred_modulation": pred_mod,
                        "count": count,
                        "total_true_in_group": total,
                        "rate_among_true": f"{(count / total if total else 0.0):.8f}",
                    }
                )
    return output_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create detailed baseline diagnostics from test predictions."
    )
    parser.add_argument(
        "--predictions-path",
        default="results/baseline/test_predictions.csv",
        help="Path to test prediction CSV.",
    )
    parser.add_argument(
        "--split-meta-path",
        default="data/splits/split_meta.json",
        help="Path to split metadata for class order.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Directory for diagnostic outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = read_predictions(args.predictions_path)
    class_order = load_class_order(args.split_meta_path, rows)

    accuracy_rows = build_accuracy_by_modulation_snr(rows, class_order)
    accuracy_path = output_dir / "test_accuracy_by_modulation_snr.csv"
    write_csv(
        accuracy_path,
        accuracy_rows,
        ["modulation", "snr", "correct", "total", "accuracy"],
    )

    grouped_rows = split_rows_by_snr_group(rows)
    for group_name, group_rows in grouped_rows.items():
        matrix = build_confusion_matrix(group_rows, class_order)
        matrix_path = output_dir / f"test_confusion_matrix_{group_name}.csv"
        normalized_path = output_dir / f"test_confusion_matrix_{group_name}_normalized.csv"
        write_matrix(matrix_path, matrix, class_order, normalized=False)
        write_matrix(normalized_path, matrix, class_order, normalized=True)
        print(f"{group_name}: rows={len(group_rows)}")
        print(f"Saved {matrix_path}")
        print(f"Saved {normalized_path}")

    top3_rows = build_top3_by_snr_group(grouped_rows, class_order)
    top3_path = output_dir / "test_top3_misclassifications_by_snr_group.csv"
    write_csv(
        top3_path,
        top3_rows,
        [
            "snr_group",
            "true_modulation",
            "rank",
            "pred_modulation",
            "count",
            "total_true_in_group",
            "rate_among_true",
        ],
    )

    print(f"Saved {accuracy_path}")
    print(f"Saved {top3_path}")


if __name__ == "__main__":
    main()
