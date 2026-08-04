from feature_probe.fitting_experiment import candidate_specs, minimum_bits_by_threshold


def test_candidate_grid_uses_published_families_without_duplicate_keys():
    specs = candidate_specs(
        ["mean_shift", "feature_re", "fitnets", "residual_adapter", "spatial_gated_fitnets"],
        [1, 2],
        [0, 1],
        fitnets_kernels=[1, 3],
        feature_re_mask_penalties=[0.0, 1e-3],
        spatial_gated_ranks=[2, 4],
    )

    assert len(specs) == 19
    assert len({spec.key for spec in specs}) == len(specs)
    assert sum(spec.family == "mean_shift" for spec in specs) == 1
    assert sum(spec.family == "spatial_gated_fitnets" for spec in specs) == 4


def test_minimum_bits_requires_error_and_functional_fidelity():
    rows = [
        {
            "candidate_key": "small-invalid",
            "family": "mean_shift",
            "rank": 0,
            "kernel": 0,
            "mask_penalty": 0.0,
            "fit_seed": 0,
            "pruning": 0.0,
            "quantization": "int4",
            "test_nrmse": 0.1,
            "fitted_asr": 0.2,
            "source_asr": 0.99,
            "functional_valid": False,
            "bits": {"total_bits": 100},
        },
        {
            "candidate_key": "valid",
            "family": "fitnets",
            "rank": 2,
            "kernel": 3,
            "mask_penalty": 0.0,
            "fit_seed": 1,
            "pruning": 0.5,
            "quantization": "int8",
            "test_nrmse": 0.15,
            "fitted_asr": 0.97,
            "source_asr": 0.99,
            "functional_valid": True,
            "bits": {"total_bits": 500},
        },
    ]

    result = minimum_bits_by_threshold(rows, [0.1, 0.2])

    assert result["0.1"] is None
    assert result["0.2"]["candidate_key"] == "valid"
