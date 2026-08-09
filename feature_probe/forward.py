from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import feature_change_summary, linear_cka, mean_cosine_distance, nearest_centroid_auc
from .split import SplitClassifier, resolve_module


SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def assert_pair_alignment(first: dict, second: dict) -> None:
    """Require two mappings to use exactly the same base examples and splits."""
    for key in ("clean", "labels", "indices", "split_codes"):
        if not torch.equal(first[key], second[key]):
            raise ValueError(f"Pair bundles are not aligned at {key!r}")


def subset_pair_bundle_by_split(bundle: dict, max_examples_per_split: int) -> dict:
    """Take the same deterministic prefix from every split of a pair bundle."""
    limit = int(max_examples_per_split)
    if limit < 1:
        raise ValueError("max_examples_per_split must be positive")
    positions = []
    for split_code in SPLIT_CODE.values():
        selected = torch.nonzero(bundle["split_codes"] == split_code, as_tuple=False).flatten()
        if selected.numel() == 0:
            raise ValueError(f"Pair bundle contains no split code {split_code}")
        positions.append(selected[:limit])
    positions = torch.cat(positions)
    return {
        key: value[positions].clone() if isinstance(value, torch.Tensor) else value
        for key, value in bundle.items()
    }


def split_mask(bundle: dict, split: str) -> torch.Tensor:
    if split not in SPLIT_CODE:
        raise ValueError(f"Unknown split: {split}")
    mask = bundle["split_codes"] == SPLIT_CODE[split]
    if not mask.any():
        raise ValueError(f"Pair bundle contains no {split} examples")
    return mask


@dataclass(frozen=True)
class FeaturePairs:
    clean: torch.Tensor
    mapped: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    split_codes: torch.Tensor

    def select(self, split: str) -> "FeaturePairs":
        mask = self.split_codes == SPLIT_CODE[split]
        if not mask.any():
            raise ValueError(f"Feature pairs contain no {split} examples")
        feature_mask = mask.to(self.clean.device)
        return FeaturePairs(
            self.clean[feature_mask],
            self.mapped[feature_mask],
            self.labels[mask],
            self.indices[mask],
            self.split_codes[mask],
        )

    def feature_tensors_to(self, device: str) -> "FeaturePairs":
        return FeaturePairs(
            self.clean.to(device),
            self.mapped.to(device),
            self.labels,
            self.indices,
            self.split_codes,
        )

    @property
    def delta(self) -> torch.Tensor:
        return self.mapped - self.clean


@torch.no_grad()
def _extract_many(model, images: torch.Tensor, layers: list[str]) -> dict[str, torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        module = resolve_module(model, layer)

        def hook(_module, _inputs, output, *, layer_name=layer):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Layer {layer_name!r} returned {type(output)!r}, expected Tensor")
            captured[layer_name].append(output.detach())

        handles.append(module.register_forward_hook(hook))
    try:
        model(images)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer, values in captured.items():
        if len(values) != 1:
            raise RuntimeError(f"Layer {layer!r} executed {len(values)} times")
        result[layer] = values[0]
    return result


@torch.no_grad()
def extract_feature_pairs_by_layer(
    model,
    bundle: dict,
    layers: list[str],
    *,
    device: str,
    batch_size: int = 128,
) -> dict[str, FeaturePairs]:
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be a non-empty list without duplicates")
    model = model.to(device).eval()
    clean_parts = {layer: [] for layer in layers}
    mapped_parts = {layer: [] for layer in layers}
    for start in range(0, len(bundle["clean"]), int(batch_size)):
        clean = bundle["clean"][start:start + int(batch_size)].to(device)
        mapped = bundle["mapped"][start:start + int(batch_size)].to(device)
        clean_features = _extract_many(model, clean, layers)
        mapped_features = _extract_many(model, mapped, layers)
        for layer in layers:
            clean_parts[layer].append(clean_features[layer].cpu())
            mapped_parts[layer].append(mapped_features[layer].cpu())
    return {
        layer: FeaturePairs(
            clean=torch.cat(clean_parts[layer]),
            mapped=torch.cat(mapped_parts[layer]),
            labels=bundle["labels"].clone(),
            indices=bundle["indices"].clone(),
            split_codes=bundle["split_codes"].clone(),
        )
        for layer in layers
    }


def input_pairs(bundle: dict) -> FeaturePairs:
    return FeaturePairs(
        clean=bundle["clean"].clone(),
        mapped=bundle["mapped"].clone(),
        labels=bundle["labels"].clone(),
        indices=bundle["indices"].clone(),
        split_codes=bundle["split_codes"].clone(),
    )


def change_maps(
    pairs: FeaturePairs,
    split: str = "test",
    *,
    max_examples: int | None = None,
) -> dict[str, torch.Tensor]:
    selected = pairs.select(split)
    if max_examples is not None:
        selected = FeaturePairs(
            selected.clean[:max_examples],
            selected.mapped[:max_examples],
            selected.labels[:max_examples],
            selected.indices[:max_examples],
            selected.split_codes[:max_examples],
        )
    delta = selected.delta.abs()
    return {
        "spatial": delta.mean((0, 1)),
        "channel": delta.mean((0, 2, 3)),
    }


def compare_feature_changes(
    trigger: FeaturePairs,
    uap: FeaturePairs,
    *,
    summary_examples: int | None = None,
    similarity_examples: int | None = None,
) -> dict:
    for key in ("labels", "indices", "split_codes"):
        if not torch.equal(getattr(trigger, key), getattr(uap, key)):
            raise ValueError(f"Feature pairs are not aligned at {key!r}")
    trigger_train = trigger.select("train").delta
    uap_train = uap.select("train").delta
    trigger_validation = trigger.select("validation").delta
    uap_validation = uap.select("validation").delta
    trigger_test = trigger.select("test").delta
    uap_test = uap.select("test").delta
    trigger_test_pairs = trigger.select("test")
    uap_test_pairs = uap.select("test")
    if summary_examples is not None:
        trigger_test_pairs = FeaturePairs(
            trigger_test_pairs.clean[:summary_examples],
            trigger_test_pairs.mapped[:summary_examples],
            trigger_test_pairs.labels[:summary_examples],
            trigger_test_pairs.indices[:summary_examples],
            trigger_test_pairs.split_codes[:summary_examples],
        )
        uap_test_pairs = FeaturePairs(
            uap_test_pairs.clean[:summary_examples],
            uap_test_pairs.mapped[:summary_examples],
            uap_test_pairs.labels[:summary_examples],
            uap_test_pairs.indices[:summary_examples],
            uap_test_pairs.split_codes[:summary_examples],
        )
    if similarity_examples is not None:
        trigger_similarity = trigger_test[:similarity_examples]
        uap_similarity = uap_test[:similarity_examples]
    else:
        trigger_similarity = trigger_test
        uap_similarity = uap_test
    return {
        "trigger_summary": feature_change_summary(trigger_test_pairs.clean, trigger_test_pairs.mapped),
        "uap_summary": feature_change_summary(uap_test_pairs.clean, uap_test_pairs.mapped),
        "validation_auc": nearest_centroid_auc(
            trigger_train,
            uap_train,
            trigger_validation,
            uap_validation,
        ),
        "test_auc": nearest_centroid_auc(trigger_train, uap_train, trigger_test, uap_test),
        "test_linear_cka": linear_cka(trigger_similarity, uap_similarity),
        "test_paired_cosine_distance": mean_cosine_distance(trigger_similarity, uap_similarity),
    }


@torch.no_grad()
def bundle_asr(model, bundle: dict, target: int, *, device: str, batch_size: int = 256) -> float:
    mask = split_mask(bundle, "test") & (bundle["labels"] != int(target))
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    successes = examples = 0
    model = model.to(device).eval()
    for start in range(0, len(selected), int(batch_size)):
        indices = selected[start:start + int(batch_size)]
        images = bundle["mapped"][indices].to(device)
        successes += (model(images).argmax(1) == int(target)).sum().item()
        examples += len(images)
    return successes / max(examples, 1)


@torch.no_grad()
def fitted_feature_asr(
    model,
    mapper,
    feature_pairs: FeaturePairs,
    contexts: torch.Tensor,
    layer: str,
    target: int,
    *,
    device: str,
    batch_size: int = 128,
) -> float:
    test = feature_pairs.select("test")
    context_mask = feature_pairs.split_codes == SPLIT_CODE["test"]
    test_contexts = contexts[context_mask]
    if len(test_contexts) != len(test.clean):
        raise ValueError("Test contexts and feature pairs are misaligned")
    adapter = SplitClassifier(model)
    mapper = mapper.to(device).eval()
    successes = examples = 0
    for start in range(0, len(test.clean), int(batch_size)):
        clean_features = test.clean[start:start + int(batch_size)].to(device)
        context = test_contexts[start:start + int(batch_size)].to(device)
        labels = test.labels[start:start + int(batch_size)]
        keep = labels != int(target)
        if not keep.any():
            continue
        predicted_features = mapper(clean_features[keep.to(clean_features.device)])
        logits = adapter.forward_from_features(
            predicted_features,
            layer,
            context=context[keep.to(device)],
        )
        successes += (logits.argmax(1) == int(target)).sum().item()
        examples += int(keep.sum())
    return successes / max(examples, 1)
