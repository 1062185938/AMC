import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_radioml_dict(path):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def build_group_table_from_meta(raw_data, meta):
    index_policy = meta.get("index_policy", {})
    modulation_order = index_policy.get("modulation_order")
    snr_order = index_policy.get("snr_order")
    if not modulation_order or not snr_order:
        raise ValueError(
            "split_meta.json must contain index_policy.modulation_order and "
            "index_policy.snr_order."
        )

    groups = []
    offset = 0
    for mod in modulation_order:
        for snr in snr_order:
            snr = int(snr)
            key = (mod, snr)
            if key not in raw_data:
                continue
            num_samples = int(raw_data[key].shape[0])
            groups.append(
                {
                    "modulation": mod,
                    "snr": snr,
                    "start_index": offset,
                    "end_index": offset + num_samples,
                    "num_samples": num_samples,
                }
            )
            offset += num_samples

    return groups


def validate_group_table(rebuilt_groups, meta_groups):
    rebuilt_by_key = {
        (group["modulation"], int(group["snr"])): group for group in rebuilt_groups
    }
    meta_by_key = {}
    for group in meta_groups:
        key = (group["modulation"], int(group["snr"]))
        if key in meta_by_key:
            raise ValueError(f"Duplicate group in split_meta.json groups: {key}")
        meta_by_key[key] = group

    rebuilt_keys = set(rebuilt_by_key)
    meta_keys = set(meta_by_key)
    if rebuilt_keys != meta_keys:
        missing_in_meta = sorted(rebuilt_keys - meta_keys)
        missing_in_rebuilt = sorted(meta_keys - rebuilt_keys)
        raise ValueError(
            "Rebuilt group_table keys do not match split_meta.json groups. "
            f"missing_in_meta={missing_in_meta}, "
            f"missing_in_rebuilt={missing_in_rebuilt}"
        )

    for key in sorted(rebuilt_keys):
        rebuilt = rebuilt_by_key[key]
        expected = meta_by_key[key]
        for field in ("start_index", "end_index", "num_samples"):
            if int(rebuilt[field]) != int(expected[field]):
                raise ValueError(
                    "Rebuilt group_table is inconsistent with split_meta.json "
                    f"for group {key}: field={field}, "
                    f"rebuilt={rebuilt[field]}, meta={expected[field]}"
                )


class RadioMLDataset(Dataset):
    def __init__(
        self,
        data_path="data/raw/RML2016.10a_dict.pkl",
        indices_path="data/splits/train_indices.npy",
        split_meta_path="data/splits/split_meta.json",
        normalize=False,
    ):
        self.data_path = Path(data_path)
        self.indices_path = Path(indices_path)
        self.split_meta_path = Path(split_meta_path)
        self.normalize = normalize

        self.raw_data = load_radioml_dict(self.data_path)
        self.indices = np.load(self.indices_path).astype(np.int64)

        with self.split_meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.groups = build_group_table_from_meta(self.raw_data, self.meta)
        validate_group_table(self.groups, self.meta["groups"])

        self.starts = np.array([group["start_index"] for group in self.groups], dtype=np.int64)
        self.ends = np.array([group["end_index"] for group in self.groups], dtype=np.int64)
        self.class_to_idx = self.meta["class_to_idx"]

    def __len__(self):
        return int(self.indices.shape[0])

    def resolve_sample_id(self, sample_id):
        sample_id = int(sample_id)
        if sample_id < 0 or sample_id >= int(self.ends[-1]):
            raise IndexError(f"sample_id out of range: {sample_id}")

        group_idx = int(np.searchsorted(self.ends, sample_id, side="right"))
        group = self.groups[group_idx]
        mod = group["modulation"]
        snr = int(group["snr"])
        local_idx = sample_id - int(group["start_index"])
        return mod, snr, local_idx

    def __getitem__(self, item):
        sample_id = int(self.indices[item])
        mod, snr, local_idx = self.resolve_sample_id(sample_id)

        x = self.raw_data[(mod, snr)][local_idx].astype(np.float32)
        if self.normalize:
            std = float(x.std())
            if std > 0:
                x = (x - float(x.mean())) / std

        return {
            "x": torch.from_numpy(x),
            "label": torch.tensor(self.class_to_idx[mod], dtype=torch.long),
            "sample_id": sample_id,
            "modulation": mod,
            "snr": snr,
            "local_idx": local_idx,
        }
