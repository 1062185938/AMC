import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.amc.data import RadioMLDataset
from src.amc.models import SimpleCNN1D

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device_arg)


def maybe_limit_dataset(dataset, max_samples, seed, name):
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError(f"--max-{name}-samples must be positive, got {max_samples}.")
    if max_samples >= len(dataset):
        print(
            f"Requested max_{name}_samples={max_samples}, "
            f"using full {name} split with {len(dataset)} samples."
        )
        return dataset

    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(dataset), size=max_samples, replace=False)
    subset_indices = np.sort(subset_indices).tolist()
    print(
        f"Using reproducible sampled {name} subset: "
        f"{max_samples}/{len(dataset)} samples, seed={seed}"
    )
    return Subset(dataset, subset_indices)


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, desc="train", leave=False) if tqdm else loader
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    iterator = tqdm(loader, desc="val", leave=False) if tqdm else loader
    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def append_train_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "best_val_accuracy",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, val_accuracy, args, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "SimpleCNN1D",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "num_classes": int(dataset.meta["num_classes"]),
            "class_to_idx": dataset.meta["class_to_idx"],
            "idx_to_class": dataset.meta["idx_to_class"],
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleCNN1D baseline.")
    parser.add_argument("--data-path", default="data/raw/RML2016.10a_dict.pkl")
    parser.add_argument("--split-meta-path", default="data/splits/split_meta.json")
    parser.add_argument("--train-indices", default="data/splits/train_indices.npy")
    parser.add_argument("--val-indices", default="data/splits/val_indices.npy")
    parser.add_argument("--checkpoint-path", default="checkpoints/simple_cnn_best.pt")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_dataset_full = RadioMLDataset(
        data_path=args.data_path,
        indices_path=args.train_indices,
        split_meta_path=args.split_meta_path,
        normalize=args.normalize,
    )
    val_dataset_full = RadioMLDataset(
        data_path=args.data_path,
        indices_path=args.val_indices,
        split_meta_path=args.split_meta_path,
        normalize=args.normalize,
    )
    train_dataset = maybe_limit_dataset(
        train_dataset_full,
        args.max_train_samples,
        args.seed,
        "train",
    )
    val_dataset = maybe_limit_dataset(
        val_dataset_full,
        args.max_val_samples,
        args.seed + 1,
        "val",
    )

    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device
    )

    model = SimpleCNN1D(
        num_classes=int(train_dataset_full.meta["num_classes"]),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = -1.0
    train_log_path = Path(args.results_dir) / "train_log.csv"
    if train_log_path.exists():
        train_log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = run_eval_epoch(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                Path(args.checkpoint_path),
                model,
                optimizer,
                epoch,
                val_accuracy,
                args,
                train_dataset_full,
            )

        append_train_log(
            train_log_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "train_accuracy": f"{train_accuracy:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "val_accuracy": f"{val_accuracy:.8f}",
                "best_val_accuracy": f"{best_val_accuracy:.8f}",
            },
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"best_val_acc={best_val_accuracy:.4f}"
        )

    print(f"Best checkpoint saved to: {args.checkpoint_path}")
    print(f"Train log saved to: {train_log_path}")


if __name__ == "__main__":
    main()
