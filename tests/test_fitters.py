import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from feature_probe.fitters import (
    DeviceTensorBatches,
    build_feature_mapping,
    evaluate_feature_mapping,
    fit_feature_mapping,
    normalized_rmse,
)


def test_device_tensor_batches_preserve_aligned_order_without_collation():
    clean = torch.arange(10).reshape(5, 2)
    target = clean + 100
    batches = DeviceTensorBatches(clean, target, batch_size=2, shuffle=False)

    clean_result = torch.cat([batch[0] for batch in batches])
    target_result = torch.cat([batch[1] for batch in batches])

    assert torch.equal(clean_result, clean)
    assert torch.equal(target_result, target)


def test_device_tensor_batches_shuffle_is_reproducible_and_aligned():
    clean = torch.arange(20).reshape(10, 2)
    target = clean + 100
    first = DeviceTensorBatches(clean, target, batch_size=3, shuffle=True, seed=7)
    second = DeviceTensorBatches(clean, target, batch_size=3, shuffle=True, seed=7)

    first_batches = list(first)
    second_batches = list(second)
    first_clean = torch.cat([batch[0] for batch in first_batches])
    first_target = torch.cat([batch[1] for batch in first_batches])
    second_clean = torch.cat([batch[0] for batch in second_batches])

    assert torch.equal(first_clean, second_clean)
    assert torch.equal(first_target, first_clean + 100)


def test_zero_feature_change_has_finite_zero_nrmse():
    clean = torch.zeros(4, 2, 2, 2)

    value = normalized_rmse(clean, clean, clean)

    assert torch.isfinite(value)
    assert value.item() == 0.0


def test_mean_shift_closed_form_recovers_constant_shift():
    clean = torch.randn(8, 2, 2, 2)
    mapped = clean + 0.25
    loader = DataLoader(TensorDataset(clean, mapped), batch_size=4)
    model = build_feature_mapping("mean_shift", (2, 2, 2))

    result = fit_feature_mapping(
        model,
        loader,
        loader,
        steps=5,
        learning_rate=0.1,
        device="cpu",
    )

    assert result.history[0]["step"] == 0
    assert result.best_validation_nrmse < 1e-6


def test_feature_re_forward_uses_mask_pattern_equation():
    model = build_feature_mapping("feature_re", (1, 2, 2), mask_penalty=1e-3)
    with torch.no_grad():
        model.mask.fill_(0.25)
        model.pattern.fill_(2.0)
    clean = torch.ones(2, 1, 2, 2)

    assert torch.allclose(model(clean), torch.full_like(clean, 1.25))
    assert model.regularization_loss().item() > 0


def test_mse_training_and_validation_stay_finite_for_tiny_change():
    torch.manual_seed(1)
    clean = torch.randn(8, 2, 2, 2)
    mapped = clean + 1e-7
    loader = DataLoader(TensorDataset(clean, mapped), batch_size=4)
    model = build_feature_mapping("residual_adapter", (2, 2, 2), rank=1)

    progress = []
    result = fit_feature_mapping(
        model,
        loader,
        loader,
        steps=5,
        learning_rate=1e-3,
        device="cpu",
        validation_interval=2,
        progress_callback=progress.append,
    )

    assert [row["step"] for row in result.history] == [1, 2, 4, 5]
    assert progress == result.history
    assert math.isfinite(result.best_validation_nrmse)
    assert math.isfinite(evaluate_feature_mapping(result.model, loader, device="cpu"))
