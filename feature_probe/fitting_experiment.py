from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSpec:
    level: str
    rank: int
    fit_seed: int

    @property
    def key(self) -> str:
        return f"{self.level}-r{self.rank}-s{self.fit_seed}"


def candidate_specs(levels: list[str], ranks: list[int], fit_seeds: list[int]) -> list[CandidateSpec]:
    specs = []
    for level in levels:
        level_ranks = [1] if level in ("C0", "C1") else ranks
        for rank in level_ranks:
            for fit_seed in fit_seeds:
                specs.append(CandidateSpec(level, int(rank), int(fit_seed)))
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
            "level": best["level"],
            "rank": best["rank"],
            "fit_seed": best["fit_seed"],
            "pruning": best["pruning"],
            "quantization": best["quantization"],
            "test_nrmse": best["test_nrmse"],
            "fitted_asr": best["fitted_asr"],
            "source_asr": best["source_asr"],
        }
    return result
