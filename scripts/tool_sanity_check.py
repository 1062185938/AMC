import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from screen_tools import ACTIONS, apply_action
from src.amc.data import RadioMLDataset


TARGET_MODULATIONS = [
    "AM-SSB",
    "PAM4",
    "QAM64",
    "QPSK",
    "WBFM",
    "QAM16",
]
TARGET_SNRS = [-8, -2, 10]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def mean_power(x):
    return float(np.mean(x[0] ** 2 + x[1] ** 2))


def signal_stats(before, after):
    mean_power_before = mean_power(before)
    mean_power_after = mean_power(after)
    std_before = float(np.std(before))
    std_after = float(np.std(after))
    eps = 1e-12
    return {
        "mean_power_before": mean_power_before,
        "mean_power_after": mean_power_after,
        "power_ratio": mean_power_after / (mean_power_before + eps),
        "max_abs_before": float(np.max(np.abs(before))),
        "max_abs_after": float(np.max(np.abs(after))),
        "std_before": std_before,
        "std_after": std_after,
        "std_ratio": std_after / (std_before + eps),
        "l2_diff": float(np.linalg.norm(after - before)),
        "has_nan": int(np.isnan(after).any()),
        "has_inf": int(np.isinf(after).any()),
    }


def complex_spectrum_db(x):
    iq = x[0].astype(np.float32) + 1j * x[1].astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(iq))
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.shape[0], d=1.0))
    return freqs, magnitude_db


def save_spectrum_figure(path, before, after, title):
    freqs_before, spectrum_before = complex_spectrum_db(before)
    freqs_after, spectrum_after = complex_spectrum_db(after)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(freqs_before, spectrum_before, linewidth=1.2)
    axes[0].set_title("Original spectrum")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs_after, spectrum_after, linewidth=1.2, color="tab:orange")
    axes[1].set_title("Processed spectrum")
    axes[1].set_xlabel("Normalized frequency")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_samples(dataset, target_modulations, target_snrs):
    target_pairs = {
        (modulation, int(snr))
        for modulation in target_modulations
        for snr in target_snrs
    }
    selected = {}

    for sample_id in dataset.indices:
        sample_id = int(sample_id)
        modulation, snr, local_idx = dataset.resolve_sample_id(sample_id)
        pair = (modulation, snr)
        if pair not in target_pairs or pair in selected:
            continue
        x = dataset.raw_data[(modulation, snr)][local_idx].astype(np.float32)
        selected[pair] = {
            "sample_id": sample_id,
            "modulation": modulation,
            "snr": snr,
            "local_idx": int(local_idx),
            "x": x,
        }
        if len(selected) == len(target_pairs):
            break

    return selected


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{float(value):.8f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check signal changes before and after fixed tool actions."
    )
    parser.add_argument("--data-path", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--split-meta-path", default="data/splits/split_meta.json")
    parser.add_argument("--val-indices", default="data/splits/val_indices.npy")
    parser.add_argument("--output-dir", default="results/tool_sanity")
    parser.add_argument("--actions", nargs="+", choices=ACTIONS, default=ACTIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    dataset = RadioMLDataset(
        data_path=args.data_path,
        indices_path=args.val_indices,
        split_meta_path=args.split_meta_path,
        normalize=False,
    )
    selected = select_samples(dataset, TARGET_MODULATIONS, TARGET_SNRS)

    missing_pairs = [
        (modulation, snr)
        for modulation in TARGET_MODULATIONS
        for snr in TARGET_SNRS
        if (modulation, snr) not in selected
    ]
    if missing_pairs:
        print(f"Warning: missing target samples: {missing_pairs}")

    rows = []
    for modulation in TARGET_MODULATIONS:
        for snr in TARGET_SNRS:
            sample = selected.get((modulation, snr))
            if sample is None:
                continue

            before = sample["x"].astype(np.float32, copy=False)
            for action in args.actions:
                after = apply_action(before.copy(), action).astype(np.float32, copy=False)
                if after.shape != before.shape:
                    raise ValueError(
                        f"Action {action} changed shape for sample "
                        f"{sample['sample_id']}: before={before.shape}, after={after.shape}"
                    )

                stats = signal_stats(before, after)
                rows.append(
                    {
                        "action": action,
                        "sample_id": sample["sample_id"],
                        "modulation": modulation,
                        "snr": snr,
                        "mean_power_before": format_float(stats["mean_power_before"]),
                        "mean_power_after": format_float(stats["mean_power_after"]),
                        "power_ratio": format_float(stats["power_ratio"]),
                        "max_abs_before": format_float(stats["max_abs_before"]),
                        "max_abs_after": format_float(stats["max_abs_after"]),
                        "std_before": format_float(stats["std_before"]),
                        "std_after": format_float(stats["std_after"]),
                        "std_ratio": format_float(stats["std_ratio"]),
                        "l2_diff": format_float(stats["l2_diff"]),
                        "has_nan": stats["has_nan"],
                        "has_inf": stats["has_inf"],
                    }
                )

                figure_name = (
                    f"{safe_name(modulation)}_snr{snr}_"
                    f"sample{sample['sample_id']}_{safe_name(action)}.png"
                )
                save_spectrum_figure(
                    figures_dir / figure_name,
                    before,
                    after,
                    title=(
                        f"{modulation}, SNR={snr}, "
                        f"sample_id={sample['sample_id']}, action={action}"
                    ),
                )

    fieldnames = [
        "action",
        "sample_id",
        "modulation",
        "snr",
        "mean_power_before",
        "mean_power_after",
        "power_ratio",
        "max_abs_before",
        "max_abs_after",
        "std_before",
        "std_after",
        "std_ratio",
        "l2_diff",
        "has_nan",
        "has_inf",
    ]
    stats_path = output_dir / "tool_sanity_stats.csv"
    write_csv(stats_path, rows, fieldnames)

    print(f"Selected samples: {len(selected)}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
