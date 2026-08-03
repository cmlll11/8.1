from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT  # noqa: F401
from feature_probe.codec import count_feature_mapping_bits
from feature_probe.fitters import build_feature_mapping, fit_feature_mapping
from feature_probe.mappings import ConstantPatch, UniversalAdditivePerturbation
from feature_probe.metrics import feature_change_summary, linear_cka, nearest_centroid_auc
from feature_probe.split import SplitClassifier


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        return self.head(x.mean((2, 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = args.device
    classifier = TinyClassifier().to(device).eval()
    split = SplitClassifier(classifier)
    images = torch.rand(32, 3, 32, 32, device=device)
    patch = ConstantPatch(top=29, left=29, size=3)
    uap = UniversalAdditivePerturbation(torch.full((3, 32, 32), 1 / 255))
    patched = patch.apply(images)
    perturbed = uap.apply(images)
    consistency = split.assert_split_consistency(images[:4], ["stem", "layer1"])
    clean_z = split.extract_features(images, "stem")
    patch_z = split.extract_features(patched, "stem")
    uap_z = split.extract_features(perturbed, "stem")
    patch_delta = patch_z - clean_z
    uap_delta = uap_z - clean_z
    auc = nearest_centroid_auc(patch_delta[:16], uap_delta[:16], patch_delta[16:], uap_delta[16:])
    dataset = TensorDataset(clean_z.detach().cpu(), patch_z.detach().cpu())
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    mapper = build_feature_mapping("residual_adapter", tuple(clean_z.shape[1:]), rank=2)
    fit = fit_feature_mapping(mapper, loader, loader, steps=args.steps, learning_rate=1e-3, device=device)
    bits = count_feature_mapping_bits(
        fit.model,
        layer_id="stem",
        family="residual_adapter",
        rank=2,
        kernel=1,
        quantization="int8",
        pruning=0.5,
    )
    output = {
        "status": "completed",
        "scientific_result": False,
        "split_max_errors": consistency,
        "patch_summary": feature_change_summary(clean_z, patch_z),
        "uap_summary": feature_change_summary(clean_z, uap_z),
        "patch_uap_linear_cka": linear_cka(patch_delta, uap_delta),
        "nearest_centroid_auc": auc,
        "fit_validation_nrmse": fit.best_validation_nrmse,
        "encoded_bits": bits.as_dict(),
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
