import torch

from feature_probe.fitting_experiment import candidate_specs, fitting_fingerprint, minimum_bits


def test_multitype_candidate_families_are_registered():
    specs = candidate_specs(
        ["mean_shift", "fitnets", "input_conditioned_fitnets", "frequency_basis_residual"],
        [1, 4], [0], fitnets_kernels=[1, 3], feature_re_mask_penalties=[0.0], spatial_gated_ranks=[8]
    )
    families = {spec.family for spec in specs}
    assert "input_conditioned_fitnets" in families
    assert "frequency_basis_residual" in families


def test_minimum_bits_uses_validation_threshold_and_activation_gate():
    rows = [
        {"candidate_key": "small", "family": "mean_shift", "rank": 0, "kernel": 0, "pruning": 0.0, "quantization": "int4", "validation_nrmse": 0.09, "test_nrmse": 0.2, "bits": {"total_bits": 100}, "activation_valid": False},
        {"candidate_key": "valid", "family": "fitnets", "rank": 1, "kernel": 1, "pruning": 0.0, "quantization": "int8", "validation_nrmse": 0.1, "test_nrmse": 0.11, "bits": {"total_bits": 120}, "activation_valid": True, "fitted_asr": 0.95, "source_asr": 0.96},
    ]
    assert minimum_bits(rows, nrmse_threshold=0.1)["candidate_key"] == "small"
    assert minimum_bits(rows, nrmse_threshold=0.1, require_activation=True)["candidate_key"] == "valid"


def test_fitting_fingerprint_changes_with_source_asset():
    config = {"fitting": {"steps": 2, "nrmse_threshold": 0.1}}
    first = fitting_fingerprint(config, protocol="MDL-FEATURE-v1", seed=0, trigger_id="badnets", condition="trigger_backdoor", layer="stem", steps=2, model_sha256="a", pair_sha256="b")
    second = fitting_fingerprint(config, protocol="MDL-FEATURE-v1", seed=0, trigger_id="badnets", condition="trigger_backdoor", layer="stem", steps=2, model_sha256="changed", pair_sha256="b")

    assert first != second
