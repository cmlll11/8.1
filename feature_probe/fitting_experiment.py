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
) -> list[CandidateSpec]:
    allowed = {"mean_shift", "feature_re", "fitnets", "residual_adapter"}
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
