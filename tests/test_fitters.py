import torch
from torch.utils.data import DataLoader, TensorDataset

from feature_probe.fitters import build_feature_mapping, fit_feature_mapping


def test_validation_interval_reduces_full_validation_passes():
    clean = torch.zeros(8, 2, 2, 2)
    mapped = clean + 0.1
    loader = DataLoader(TensorDataset(clean, mapped), batch_size=4)
    model = build_feature_mapping("C1", (2, 2, 2))

    result = fit_feature_mapping(
        model,
        loader,
        loader,
        steps=5,
        learning_rate=0.1,
        device="cpu",
        validation_interval=2,
    )

    assert [row["step"] for row in result.history] == [1, 2, 4, 5]
    assert result.best_validation_nrmse >= 0
