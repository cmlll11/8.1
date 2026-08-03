import torch
from torch.utils.data import DataLoader, TensorDataset

from feature_probe.fitters import DeviceTensorBatches, build_feature_mapping, fit_feature_mapping


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


def test_validation_interval_reduces_full_validation_passes():
    clean = torch.zeros(8, 2, 2, 2)
    mapped = clean + 0.1
    loader = DataLoader(TensorDataset(clean, mapped), batch_size=4)
    model = build_feature_mapping("C1", (2, 2, 2))

    progress = []
    result = fit_feature_mapping(
        model,
        loader,
        loader,
        steps=5,
        learning_rate=0.1,
        device="cpu",
        validation_interval=2,
        progress_callback=progress.append,
    )

    assert [row["step"] for row in result.history] == [1, 2, 4, 5]
    assert progress == result.history
    assert result.best_validation_nrmse >= 0
