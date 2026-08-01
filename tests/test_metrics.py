import torch

from feature_probe.metrics import binary_roc_auc, feature_change_summary, linear_cka


def test_auc_perfect_order():
    assert binary_roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])) == 1.0


def test_summary_and_cka_identity():
    torch.manual_seed(0)
    clean = torch.rand(5, 3, 4, 4)
    delta = torch.rand_like(clean) * 0.1
    summary = feature_change_summary(clean, clean + delta)
    assert summary["relative_l2"] > 0
    assert abs(linear_cka(delta, delta) - 1.0) < 1e-9
