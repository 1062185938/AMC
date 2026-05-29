import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


SNR_GROUPS = [
    ("Hard", lambda snr: snr <= -10),
    ("Medium", lambda snr: -8 <= snr <= -2),
    ("Easy", lambda snr: snr >= 0),
]


def snr_group_name(snr):
    for name, predicate in SNR_GROUPS:
        if predicate(snr):
            return name
    raise ValueError(f"SNR does not belong to any configured group: {snr}")


def read_predictions(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "action",
            "sample_id",
            "true_modulation",
            "true_snr",
            "correct",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {path}: {sorted(missing_fields)}"
            )
        return list(reader)


def init_counter():
    return {"correct": 0, "total": 0}


def update_counter(counter, correct):
    counter["correct"] += int(correct)
    counter["total"] += 1


def accuracy(counter):
    total = counter["total"]
    return counter["correct"] / total if total else 0.0


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_orders(rows):
    action_order = OrderedDict()
    modulation_order = OrderedDict()
    for row in rows:
        action_order.setdefault(row["action"], None)
        modulation_order.setdefault(row["true_modulation"], None)
    return list(action_order.keys()), list(modulation_order.keys())


def build_no_process_maps(rows):
    no_process_by_sample = {}
    no_process_counters = defaultdict(init_counter)

    for row in rows:
        if row["action"] != "no_process":
            continue
        sample_id = int(row["sample_id"])
        modulation = row["true_modulation"]
        snr = int(row["true_snr"])
        group = snr_group_name(snr)
        correct = int(row["correct"])
        key = (modulation, group)

        if sample_id in no_process_by_sample:
            raise ValueError(f"Duplicate no_process sample_id: {sample_id}")
        no_process_by_sample[sample_id] = {
            "correct": bool(correct),
            "true_modulation": modulation,
            "snr_group": group,
        }
        update_counter(no_process_counters[key], correct)

    if not no_process_by_sample:
        raise ValueError("No no_process rows found in predictions_by_action.csv.")

    return no_process_by_sample, no_process_counters


def analyze_gains(rows):
    action_order, modulation_order = collect_orders(rows)
    no_process_by_sample, no_process_counters = build_no_process_maps(rows)

    action_counters = defaultdict(init_counter)
    gain_counters = defaultdict(lambda: {"error_to_correct": 0, "correct_to_error": 0})

    for row in rows:
        action = row["action"]
        sample_id = int(row["sample_id"])
        modulation = row["true_modulation"]
        snr_group = snr_group_name(int(row["true_snr"]))
        correct = bool(int(row["correct"]))

        baseline = no_process_by_sample.get(sample_id)
        if baseline is None:
            raise ValueError(f"Missing no_process baseline for sample_id={sample_id}")
        if baseline["true_modulation"] != modulation or baseline["snr_group"] != snr_group:
            raise ValueError(
                "Inconsistent sample metadata between action and no_process: "
                f"sample_id={sample_id}"
            )

        key = (action, modulation, snr_group)
        update_counter(action_counters[key], correct)

        baseline_correct = baseline["correct"]
        if not baseline_correct and correct:
            gain_counters[key]["error_to_correct"] += 1
        elif baseline_correct and not correct:
            gain_counters[key]["correct_to_error"] += 1

    rows_out = []
    snr_group_order = [name for name, _ in SNR_GROUPS]
    for action in action_order:
        for modulation in modulation_order:
            for snr_group in snr_group_order:
                group_key = (modulation, snr_group)
                action_key = (action, modulation, snr_group)
                no_process_counter = no_process_counters[group_key]
                action_counter = action_counters[action_key]
                gain_counter = gain_counters[action_key]

                no_process_accuracy = accuracy(no_process_counter)
                action_accuracy = accuracy(action_counter)
                error_to_correct = gain_counter["error_to_correct"]
                correct_to_error = gain_counter["correct_to_error"]
                rows_out.append(
                    {
                        "action": action,
                        "true_modulation": modulation,
                        "snr_group": snr_group,
                        "no_process_accuracy": f"{no_process_accuracy:.8f}",
                        "action_accuracy": f"{action_accuracy:.8f}",
                        "delta_accuracy": f"{action_accuracy - no_process_accuracy:.8f}",
                        "error_to_correct": error_to_correct,
                        "correct_to_error": correct_to_error,
                        "net_gain": error_to_correct - correct_to_error,
                        "total_samples": action_counter["total"],
                    }
                )

    return rows_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze tool gains from predictions_by_action.csv without rerunning models."
    )
    parser.add_argument(
        "--predictions-path",
        default="results/tool_screening/predictions_by_action.csv",
        help="Path to predictions_by_action.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tool_analysis",
        help="Directory for tool gain analysis outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_predictions(args.predictions_path)
    gain_rows = analyze_gains(rows)

    output_path = Path(args.output_dir) / "tool_gain_by_modulation_snr_group.csv"
    write_csv(
        output_path,
        gain_rows,
        [
            "action",
            "true_modulation",
            "snr_group",
            "no_process_accuracy",
            "action_accuracy",
            "delta_accuracy",
            "error_to_correct",
            "correct_to_error",
            "net_gain",
            "total_samples",
        ],
    )
    print(f"Saved tool gain analysis: {output_path}")


if __name__ == "__main__":
    main()
