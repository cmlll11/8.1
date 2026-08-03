from feature_probe.fitting_experiment import candidate_specs, minimum_bits_by_threshold


def test_candidate_grid_does_not_repeat_rank_for_c0_or_c1():
    specs = candidate_specs(["C0", "C1", "C2"], [1, 2, 4], [0, 1])

    assert len(specs) == 10
    assert len({spec.key for spec in specs}) == len(specs)


def test_minimum_bits_requires_error_and_functional_fidelity():
    rows = [
        {
            "candidate_key": "small-invalid",
            "level": "C0",
            "rank": 1,
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
            "level": "C2",
            "rank": 2,
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
