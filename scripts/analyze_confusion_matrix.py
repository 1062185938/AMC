import argparse
import csv
import json
from collections import Counter
from pathlib import Path


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


def read_predictions(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"true_modulation", "pred_modulation"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {path}: {sorted(missing_fields)}"
            )
        return list(reader)


def build_confusion_matrix(rows, class_order):
    matrix = {
        true_mod: {pred_mod: 0 for pred_mod in class_order}
        for true_mod in class_order
    }
    for row in rows:
        true_mod = row["true_modulation"]
        pred_mod = row["pred_modulation"]
        if true_mod not in matrix:
            matrix[true_mod] = {mod: 0 for mod in class_order}
        if pred_mod not in matrix[true_mod]:
            for existing_row in matrix.values():
                existing_row[pred_mod] = 0
            class_order.append(pred_mod)
        matrix[true_mod][pred_mod] += 1
    return matrix


def write_matrix(path, matrix, class_order, normalized=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["true_modulation"] + class_order
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
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
            writer.writerow(row)


def build_top3_confusions(matrix, class_order):
    rows = []
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
            rows.append(
                {
                    "true_modulation": true_mod,
                    "rank": rank,
                    "pred_modulation": pred_mod,
                    "count": count,
                    "rate_among_true": f"{(count / total if total else 0.0):.8f}",
                }
            )
    return rows


def write_top3(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "true_modulation",
        "rank",
        "pred_modulation",
        "count",
        "rate_among_true",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top3(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["true_modulation"], []).append(row)

    print("Top-3 misclassifications by true modulation:")
    for true_mod in grouped:
        parts = [
            f"{row['rank']}. {row['pred_modulation']} "
            f"count={row['count']} rate={row['rate_among_true']}"
            for row in grouped[true_mod]
        ]
        print(f"{true_mod}: " + "; ".join(parts))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze test prediction confusion matrix."
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
        help="Directory for confusion matrix outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows = read_predictions(args.predictions_path)
    class_order = load_class_order(args.split_meta_path, rows)
    matrix = build_confusion_matrix(rows, class_order)
    top3_rows = build_top3_confusions(matrix, class_order)

    matrix_path = output_dir / "test_confusion_matrix.csv"
    normalized_path = output_dir / "test_confusion_matrix_normalized.csv"
    top3_path = output_dir / "test_top3_misclassifications.csv"

    write_matrix(matrix_path, matrix, class_order, normalized=False)
    write_matrix(normalized_path, matrix, class_order, normalized=True)
    write_top3(top3_path, top3_rows)

    print(f"Saved confusion matrix: {matrix_path}")
    print(f"Saved normalized confusion matrix: {normalized_path}")
    print(f"Saved top-3 misclassifications: {top3_path}")
    print_top3(top3_rows)


if __name__ == "__main__":
    main()
