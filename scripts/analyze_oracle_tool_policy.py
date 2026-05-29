import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


EXCLUDED_ACTIONS = {"normalize_power"}
SNR_GROUPS = [
    ("Hard", lambda snr: snr <= -10),
    ("Medium", lambda snr: -8 <= snr <= -2),
    ("Easy", lambda snr: snr >= 0),
]
ACTION_PRIORITY = {
    "no_process": 0,
    "wavelet_weak": 1,
    "wavelet_strong": 2,
    "lowpass_mild": 3,
    "lowpass_strong": 4,
}


def snr_group_name(snr):
    for name, predicate in SNR_GROUPS:
        if predicate(snr):
            return name
    raise ValueError(f"SNR does not belong to any configured group: {snr}")


def read_csv(path, required_fields):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_fields = set(required_fields) - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {path}: {sorted(missing_fields)}"
            )
        return list(reader)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_modulation_order(prediction_rows):
    order = OrderedDict()
    for row in prediction_rows:
        order.setdefault(row["true_modulation"], None)
    return list(order.keys())


def build_oracle_policy(gain_rows):
    candidates_by_group = defaultdict(list)
    for row in gain_rows:
        action = row["action"]
        if action in EXCLUDED_ACTIONS:
            continue
        key = (row["true_modulation"], row["snr_group"])
        candidates_by_group[key].append(row)

    policy = {}
    policy_rows = []
    for key in sorted(candidates_by_group):
        candidates = candidates_by_group[key]
        candidates.sort(
            key=lambda row: (
                -float(row["action_accuracy"]),
                -int(row["net_gain"]),
                ACTION_PRIORITY.get(row["action"], 999),
            )
        )
        best = candidates[0]
        true_modulation, snr_group = key
        policy[key] = best["action"]
        policy_rows.append(
            {
                "true_modulation": true_modulation,
                "snr_group": snr_group,
                "best_action": best["action"],
                "no_process_accuracy": best["no_process_accuracy"],
                "best_action_accuracy": best["action_accuracy"],
                "delta_accuracy": best["delta_accuracy"],
                "net_gain": best["net_gain"],
                "total_samples": best["total_samples"],
            }
        )

    return policy, policy_rows


def build_prediction_maps(prediction_rows):
    by_sample_action = {}
    no_process_by_sample = {}

    for row in prediction_rows:
        action = row["action"]
        sample_id = int(row["sample_id"])
        key = (sample_id, action)
        if key in by_sample_action:
            raise ValueError(f"Duplicate prediction row for sample/action: {key}")
        by_sample_action[key] = row

        if action == "no_process":
            if sample_id in no_process_by_sample:
                raise ValueError(f"Duplicate no_process row for sample_id={sample_id}")
            no_process_by_sample[sample_id] = row

    if not no_process_by_sample:
        raise ValueError("No no_process rows found in predictions_by_action.csv.")

    return by_sample_action, no_process_by_sample


def init_counter():
    return {"correct": 0, "total": 0}


def update_counter(counter, correct):
    counter["correct"] += int(correct)
    counter["total"] += 1


def accuracy(counter):
    return counter["correct"] / counter["total"] if counter["total"] else 0.0


def select_oracle_predictions(prediction_rows, policy):
    by_sample_action, no_process_by_sample = build_prediction_maps(prediction_rows)
    oracle_rows = []

    baseline_counter = init_counter()
    oracle_counter = init_counter()
    baseline_by_snr_group = defaultdict(init_counter)
    oracle_by_snr_group = defaultdict(init_counter)
    baseline_by_modulation = defaultdict(init_counter)
    oracle_by_modulation = defaultdict(init_counter)
    error_to_correct = 0
    correct_to_error = 0

    for sample_id in sorted(no_process_by_sample):
        baseline_row = no_process_by_sample[sample_id]
        modulation = baseline_row["true_modulation"]
        snr = int(baseline_row["true_snr"])
        group = snr_group_name(snr)
        policy_key = (modulation, group)
        if policy_key not in policy:
            raise ValueError(f"Missing oracle policy for group: {policy_key}")

        selected_action = policy[policy_key]
        selected_row = by_sample_action.get((sample_id, selected_action))
        if selected_row is None:
            raise ValueError(
                f"Missing prediction for sample_id={sample_id}, action={selected_action}"
            )

        baseline_correct = bool(int(baseline_row["correct"]))
        oracle_correct = bool(int(selected_row["correct"]))

        update_counter(baseline_counter, baseline_correct)
        update_counter(oracle_counter, oracle_correct)
        update_counter(baseline_by_snr_group[group], baseline_correct)
        update_counter(oracle_by_snr_group[group], oracle_correct)
        update_counter(baseline_by_modulation[modulation], baseline_correct)
        update_counter(oracle_by_modulation[modulation], oracle_correct)

        if not baseline_correct and oracle_correct:
            error_to_correct += 1
        elif baseline_correct and not oracle_correct:
            correct_to_error += 1

        oracle_rows.append(
            {
                "selected_action": selected_action,
                "sample_id": sample_id,
                "true_label": selected_row["true_label"],
                "true_modulation": modulation,
                "true_snr": snr,
                "snr_group": group,
                "pred_label": selected_row["pred_label"],
                "pred_modulation": selected_row["pred_modulation"],
                "correct": int(oracle_correct),
                "top1_confidence": selected_row["top1_confidence"],
                "top2_confidence": selected_row["top2_confidence"],
                "confidence_margin": selected_row["confidence_margin"],
                "entropy": selected_row["entropy"],
                "no_process_correct": int(baseline_correct),
            }
        )

    counters = {
        "baseline": baseline_counter,
        "oracle": oracle_counter,
        "baseline_by_snr_group": baseline_by_snr_group,
        "oracle_by_snr_group": oracle_by_snr_group,
        "baseline_by_modulation": baseline_by_modulation,
        "oracle_by_modulation": oracle_by_modulation,
        "error_to_correct": error_to_correct,
        "correct_to_error": correct_to_error,
    }
    return oracle_rows, counters


def build_summary_rows(counters):
    baseline = counters["baseline"]
    oracle = counters["oracle"]
    baseline_accuracy = accuracy(baseline)
    oracle_accuracy = accuracy(oracle)
    error_to_correct = counters["error_to_correct"]
    correct_to_error = counters["correct_to_error"]
    return [
        {
            "baseline_action": "no_process",
            "oracle_policy": "best_action_by_true_modulation_snr_group",
            "baseline_accuracy": f"{baseline_accuracy:.8f}",
            "oracle_accuracy": f"{oracle_accuracy:.8f}",
            "delta_accuracy": f"{oracle_accuracy - baseline_accuracy:.8f}",
            "baseline_correct": baseline["correct"],
            "oracle_correct": oracle["correct"],
            "total_samples": oracle["total"],
            "error_to_correct": error_to_correct,
            "correct_to_error": correct_to_error,
            "net_gain": error_to_correct - correct_to_error,
            "note": "Oracle upper bound; not a deployable method.",
        }
    ]


def build_group_rows(counters, group_type, group_order):
    baseline_key = f"baseline_by_{group_type}"
    oracle_key = f"oracle_by_{group_type}"
    id_field = group_type
    rows = []
    for group in group_order:
        baseline = counters[baseline_key][group]
        oracle = counters[oracle_key][group]
        baseline_accuracy = accuracy(baseline)
        oracle_accuracy = accuracy(oracle)
        rows.append(
            {
                id_field: group,
                "baseline_correct": baseline["correct"],
                "oracle_correct": oracle["correct"],
                "total_samples": oracle["total"],
                "baseline_accuracy": f"{baseline_accuracy:.8f}",
                "oracle_accuracy": f"{oracle_accuracy:.8f}",
                "delta_accuracy": f"{oracle_accuracy - baseline_accuracy:.8f}",
                "net_gain": oracle["correct"] - baseline["correct"],
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an oracle tool-selection upper bound from saved tool predictions."
    )
    parser.add_argument(
        "--gain-path",
        default="results/tool_analysis/tool_gain_by_modulation_snr_group.csv",
    )
    parser.add_argument(
        "--predictions-path",
        default="results/tool_screening/predictions_by_action.csv",
    )
    parser.add_argument("--output-dir", default="results/tool_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    gain_rows = read_csv(
        args.gain_path,
        {
            "action",
            "true_modulation",
            "snr_group",
            "no_process_accuracy",
            "action_accuracy",
            "delta_accuracy",
            "net_gain",
            "total_samples",
        },
    )
    prediction_rows = read_csv(
        args.predictions_path,
        {
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
        },
    )

    output_dir = Path(args.output_dir)
    policy, policy_rows = build_oracle_policy(gain_rows)
    oracle_rows, counters = select_oracle_predictions(prediction_rows, policy)
    summary_rows = build_summary_rows(counters)

    snr_group_order = [name for name, _ in SNR_GROUPS]
    modulation_order = collect_modulation_order(prediction_rows)
    snr_group_rows = build_group_rows(counters, "snr_group", snr_group_order)
    modulation_rows = build_group_rows(counters, "modulation", modulation_order)

    write_csv(
        output_dir / "oracle_policy_by_modulation_snr_group.csv",
        policy_rows,
        [
            "true_modulation",
            "snr_group",
            "best_action",
            "no_process_accuracy",
            "best_action_accuracy",
            "delta_accuracy",
            "net_gain",
            "total_samples",
        ],
    )
    write_csv(
        output_dir / "oracle_predictions.csv",
        oracle_rows,
        [
            "selected_action",
            "sample_id",
            "true_label",
            "true_modulation",
            "true_snr",
            "snr_group",
            "pred_label",
            "pred_modulation",
            "correct",
            "top1_confidence",
            "top2_confidence",
            "confidence_margin",
            "entropy",
            "no_process_correct",
        ],
    )
    write_csv(
        output_dir / "oracle_metrics_summary.csv",
        summary_rows,
        [
            "baseline_action",
            "oracle_policy",
            "baseline_accuracy",
            "oracle_accuracy",
            "delta_accuracy",
            "baseline_correct",
            "oracle_correct",
            "total_samples",
            "error_to_correct",
            "correct_to_error",
            "net_gain",
            "note",
        ],
    )
    write_csv(
        output_dir / "oracle_accuracy_by_snr_group.csv",
        snr_group_rows,
        [
            "snr_group",
            "baseline_correct",
            "oracle_correct",
            "total_samples",
            "baseline_accuracy",
            "oracle_accuracy",
            "delta_accuracy",
            "net_gain",
        ],
    )
    write_csv(
        output_dir / "oracle_accuracy_by_modulation.csv",
        modulation_rows,
        [
            "modulation",
            "baseline_correct",
            "oracle_correct",
            "total_samples",
            "baseline_accuracy",
            "oracle_accuracy",
            "delta_accuracy",
            "net_gain",
        ],
    )

    summary = summary_rows[0]
    print("Baseline vs Oracle:")
    print(
        f"no_process accuracy={summary['baseline_accuracy']} "
        f"oracle accuracy={summary['oracle_accuracy']} "
        f"delta={summary['delta_accuracy']} "
        f"net_gain={summary['net_gain']}"
    )
    print(f"Saved oracle outputs to: {output_dir}")


if __name__ == "__main__":
    main()
