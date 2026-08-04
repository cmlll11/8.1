from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    rank: int
    kernel: int
    mask_penalty: float
    fit_seed: int

    @property
    def key(self) -> str:
        if self.family == "mean_shift":
            return "mean_shift"
        if self.family == "feature_re":
            penalty = format(self.mask_penalty, ".0e").replace("+", "")
            return f"feature_re-lambda{penalty}-s{self.fit_seed}"
        if self.family == "fitnets":
            return f"fitnets-k{self.kernel}-r{self.rank}-s{self.fit_seed}"
        if self.family == "spatial_gated_fitnets":
            return f"spatial_gated_fitnets-r{self.rank}-s{self.fit_seed}"
        if self.family == "input_conditioned_fitnets":
            return f"input_conditioned_fitnets-k{self.kernel}-r{self.rank}-s{self.fit_seed}"
        if self.family == "frequency_basis_residual":
            return f"frequency_basis_residual-r{self.rank}-s{self.fit_seed}"
        return f"{self.family}-r{self.rank}-s{self.fit_seed}"

    def structure(self) -> dict[str, int | float | str]:
        return {
            "family": self.family,
            "rank": self.rank,
            "kernel": self.kernel,
            "mask_penalty": self.mask_penalty,
        }


def candidate_specs(
    families: list[str],
    ranks: list[int],
    fit_seeds: list[int],
    *,
    fitnets_kernels: list[int],
    feature_re_mask_penalties: list[float],
    spatial_gated_ranks: list[int],
) -> list[CandidateSpec]:
    allowed = {
        "mean_shift",
        "feature_re",
        "fitnets",
        "residual_adapter",
        "spatial_gated_fitnets",
        "input_conditioned_fitnets",
        "frequency_basis_residual",
    }
    unknown = set(families) - allowed
    if unknown:
        raise ValueError(f"Unknown feature fitting families: {sorted(unknown)}")
    if any(int(rank) < 1 for rank in ranks):
        raise ValueError("All ranks must be positive")
    specs = []
    for family in families:
        if family == "mean_shift":
            specs.append(CandidateSpec(family, 0, 0, 0.0, 0))
        elif family == "feature_re":
            for penalty in feature_re_mask_penalties:
                for fit_seed in fit_seeds:
                    specs.append(CandidateSpec(family, 0, 0, float(penalty), int(fit_seed)))
        elif family == "fitnets":
            for kernel in fitnets_kernels:
                if int(kernel) not in (1, 3):
                    raise ValueError("FitNets kernels must be 1 or 3")
                for rank in ranks:
                    for fit_seed in fit_seeds:
                        specs.append(CandidateSpec(family, int(rank), int(kernel), 0.0, int(fit_seed)))
        elif family == "residual_adapter":
            for fit_seed in fit_seeds:
                specs.append(CandidateSpec(family, 0, 1, 0.0, int(fit_seed)))
        elif family == "spatial_gated_fitnets":
            if any(int(rank) < 1 for rank in spatial_gated_ranks):
                raise ValueError("All spatial-gated ranks must be positive")
            for rank in spatial_gated_ranks:
                for fit_seed in fit_seeds:
                    specs.append(CandidateSpec(family, int(rank), 3, 0.0, int(fit_seed)))
        elif family == "input_conditioned_fitnets":
            for kernel in fitnets_kernels:
                if int(kernel) not in (1, 3):
                    raise ValueError("Input-conditioned kernels must be 1 or 3")
                for rank in ranks:
                    for fit_seed in fit_seeds:
                        specs.append(CandidateSpec(family, int(rank), int(kernel), 0.0, int(fit_seed)))
        elif family == "frequency_basis_residual":
            for rank in ranks:
                for fit_seed in fit_seeds:
                    specs.append(CandidateSpec(family, int(rank), 1, 0.0, int(fit_seed)))
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("Feature fitting candidate keys are not unique")
    return specs


def minimum_bits_by_threshold(rows: list[dict], thresholds: list[float]) -> dict[str, dict | None]:
    result: dict[str, dict | None] = {}
    for threshold in thresholds:
        valid = [
            row for row in rows
            if row["test_nrmse"] <= float(threshold) and row["functional_valid"]
        ]
        if not valid:
            result[str(float(threshold))] = None
            continue
        best = min(valid, key=lambda row: (row["bits"]["total_bits"], row["test_nrmse"]))
        result[str(float(threshold))] = {
            "total_bits": best["bits"]["total_bits"],
            "candidate_key": best["candidate_key"],
            "family": best["family"],
            "rank": best["rank"],
            "kernel": best["kernel"],
            "mask_penalty": best["mask_penalty"],
            "fit_seed": best["fit_seed"],
            "pruning": best["pruning"],
            "quantization": best["quantization"],
            "test_nrmse": best["test_nrmse"],
            "fitted_asr": best["fitted_asr"],
            "source_asr": best["source_asr"],
        }
    return result


def minimum_bits(rows: list[dict], *, nrmse_threshold: float, require_activation: bool = False) -> dict | None:
    """Select the minimum MDL candidate using validation-qualified rows.

    Rows produced by the multitype runner carry validation and test metrics.
    Selection is deliberately validation-only; test values are reported after
    the candidate has been selected.
    """
    valid = [
        row for row in rows
        if float(row.get("validation_nrmse", float("inf"))) <= float(nrmse_threshold)
        and (not require_activation or bool(row.get("activation_valid", False)))
    ]
    if not valid:
        return None
    best = min(valid, key=lambda row: (int(row["bits"]["total_bits"]), float(row["validation_nrmse"]), float(row.get("test_nrmse", float("inf")))))
    return {
        "total_bits": int(best["bits"]["total_bits"]),
        "candidate_key": best["candidate_key"],
        "family": best["family"],
        "rank": best["rank"],
        "kernel": best["kernel"],
        "pruning": best["pruning"],
        "quantization": best["quantization"],
        "validation_nrmse": best["validation_nrmse"],
        "test_nrmse": best["test_nrmse"],
        "source_asr": best.get("source_asr"),
        "fitted_asr": best.get("fitted_asr"),
        "activation_valid": bool(best.get("activation_valid", False)),
    }
