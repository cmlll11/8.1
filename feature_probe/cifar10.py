from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_FOLDER = "cifar-10-batches-py"
CIFAR10_BATCH_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class Cifar10Splits:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    train_indices: torch.Tensor
    validation_images: torch.Tensor
    validation_labels: torch.Tensor
    validation_indices: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    test_indices: torch.Tensor


def md5sum(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - CIFAR-10 publishes MD5 checksums
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_files(folder: Path) -> list[Path]:
    return [folder / f"data_batch_{index}" for index in range(1, 6)] + [folder / "test_batch"]


def ensure_cifar10(root: str | Path, *, download: bool = False) -> Path:
    root = Path(root)
    folder = root / CIFAR10_FOLDER
    if all(path.is_file() for path in _required_files(folder)):
        invalid = [name for name, checksum in CIFAR10_BATCH_MD5.items() if md5sum(folder / name) != checksum]
        if invalid:
            raise ValueError(f"CIFAR-10 extracted batch checksum mismatch: {invalid}")
        return folder
    archive = root / CIFAR10_ARCHIVE
    if not archive.is_file() and download:
        root.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".tmp")
        urllib.request.urlretrieve(CIFAR10_URL, temporary)
        temporary.replace(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"CIFAR-10 is missing under {root}; pass --download once")
    if md5sum(archive) != CIFAR10_ARCHIVE_MD5:
        raise ValueError(f"CIFAR-10 archive checksum mismatch: {archive}")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if target != resolved and resolved not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in CIFAR-10 archive: {member.name}")
        handle.extractall(root, members=members)
    return folder


def _read_batch(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")  # noqa: S301 - verified official data
    data = payload.get(b"data")
    labels = payload.get(b"labels")
    if data is None or labels is None:
        raise ValueError(f"Malformed CIFAR-10 batch: {path}")
    images = torch.from_numpy(np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)).float().div_(255)
    return images, torch.tensor(labels, dtype=torch.long)


def fixed_split_indices(total: int, validation: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < validation < total:
        raise ValueError("validation must be between zero and total")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(int(seed)))
    return order[validation:], order[:validation]


def load_cifar10(root: str | Path, *, split_seed: int = 2026, download: bool = False) -> Cifar10Splits:
    folder = ensure_cifar10(root, download=download)
    train_parts = [_read_batch(folder / f"data_batch_{index}") for index in range(1, 6)]
    images = torch.cat([part[0] for part in train_parts])
    labels = torch.cat([part[1] for part in train_parts])
    test_images, test_labels = _read_batch(folder / "test_batch")
    train_idx, validation_idx = fixed_split_indices(len(images), 5000, split_seed)
    return Cifar10Splits(
        images[train_idx], labels[train_idx], train_idx,
        images[validation_idx], labels[validation_idx], validation_idx,
        test_images, test_labels, torch.arange(len(test_images)),
    )


def normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def augment_batch(images: torch.Tensor, *, seed: int, padding: int = 4) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    padded = torch.nn.functional.pad(images, (padding, padding, padding, padding), mode="reflect")
    top = torch.randint(0, 2 * padding + 1, (len(images),), generator=generator)
    left = torch.randint(0, 2 * padding + 1, (len(images),), generator=generator)
    flips = torch.rand(len(images), generator=generator) < 0.5
    batch = torch.arange(len(images))[:, None, None]
    rows = top[:, None, None] + torch.arange(32)[None, :, None]
    columns = left[:, None, None] + torch.arange(32)[None, None, :]
    crops = padded.permute(0, 2, 3, 1)[batch, rows, columns].permute(0, 3, 1, 2)
    return torch.where(flips[:, None, None, None], torch.flip(crops, (-1,)), crops)


def select_non_target(images, labels, indices, *, target: int, count: int, seed: int):
    eligible = torch.nonzero(labels != int(target), as_tuple=False).flatten()
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(int(seed)))
    selected = eligible[order[: min(int(count), len(eligible))]]
    return images[selected], labels[selected], indices[selected]


def select_poison_indices(labels: torch.Tensor, *, target: int, fraction: float, seed: int) -> torch.Tensor:
    if not 0 <= float(fraction) <= 1:
        raise ValueError("poison fraction must be in [0, 1]")
    eligible = torch.nonzero(labels != int(target), as_tuple=False).flatten()
    count = int(round(len(eligible) * float(fraction)))
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(int(seed)))
    return eligible[order[:count]].sort().values
