from __future__ import annotations

from pathlib import Path

import torch

from .cifar10 import select_non_target


SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


@torch.no_grad()
def build_pair_bundle(splits, mapping, *, target: int, count_per_split: int, seed: int, device: str, metadata: dict):
    clean_parts = []
    mapped_parts = []
    label_parts = []
    index_parts = []
    split_parts = []
    for offset, name in enumerate(("train", "validation", "test")):
        images = getattr(splits, f"{name}_images")
        labels = getattr(splits, f"{name}_labels")
        indices = getattr(splits, f"{name}_indices")
        clean, chosen_labels, chosen_indices = select_non_target(
            images, labels, indices, target=target, count=count_per_split, seed=int(seed) + offset
        )
        generated = []
        for start in range(0, len(clean), 256):
            generated.append(mapping(clean[start:start + 256].to(device)).cpu())
        clean_parts.append(clean)
        mapped_parts.append(torch.cat(generated))
        label_parts.append(chosen_labels)
        index_parts.append(chosen_indices)
        split_parts.append(torch.full((len(clean),), SPLIT_CODE[name], dtype=torch.uint8))
    return {
        "clean": torch.cat(clean_parts),
        "mapped": torch.cat(mapped_parts),
        "labels": torch.cat(label_parts),
        "indices": torch.cat(index_parts),
        "split_codes": torch.cat(split_parts),
        "metadata": metadata,
    }


def save_pair_bundle(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
