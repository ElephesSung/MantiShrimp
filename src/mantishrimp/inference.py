"""Bayesian count-model inference for MantiShrimp simulation outputs.

The four model family and SMC workflow are adapted from the Orca inference
core supplied with the project.  The observable is deliberately generic:
the same models can analyse proximity contacts, bound synapses, or target
kills per killer cell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .results import Observable, SimulationResult

MODEL_ORDER: tuple[str, ...] = ("homo", "Z2P", "Dis2P", "hetero3")
MODEL_LABELS: Mapping[str, str] = {
    "homo": "Homogeneous Poisson",
    "Z2P": "Zero-Inflated Poisson",
    "Dis2P": "Gamma-Poisson",
    "hetero3": "Zero-Inflated Gamma-Poisson",
}


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fitted Bayesian count model."""

    model_name: str
    model_label: str
    model: pm.Model
    idata: az.InferenceData
    log_evidence: float
    observable: str | None = None


def _validate_counts(
    counts_per_cell: Sequence[int],
    exposure: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(counts_per_cell)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("counts_per_cell must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError("counts_per_cell must contain finite non-negative integers")
    counts = raw.astype(np.int64)

    if np.isscalar(exposure):
        exposure_array = np.full(counts.shape, float(exposure), dtype=float)
    else:
        exposure_array = np.asarray(exposure, dtype=float)
    if exposure_array.shape != counts.shape:
        raise ValueError("exposure must be scalar or aligned with counts_per_cell")
    if np.any(~np.isfinite(exposure_array)) or np.any(exposure_array <= 0):
        raise ValueError("exposure must be finite and strictly positive")
    return counts, exposure_array


def _resolve_cores(cores: int | None) -> int:
    available = os.cpu_count() or 1
    if cores is None or int(cores) <= 0:
        return available
    return min(int(cores), available)


def _parse_log_marginal_likelihood(
    raw,
    *,
    chains: int,
) -> np.ndarray | None:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except (TypeError, ValueError):
            return None

    if all(not isinstance(item, (list, tuple, np.ndarray)) for item in raw):
        values = np.asarray(raw, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return np.full(chains, float(values[-1]), dtype=float)
        return None

    parsed: list[float] = []
    for item in raw:
        try:
            values = np.asarray(item, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        values = values[np.isfinite(values)]
        if values.size:
            parsed.append(float(values[-1]))
    if not parsed:
        return None
    output = np.asarray(parsed, dtype=float)
    if output.size < chains:
        output = np.concatenate([output, np.full(chains - output.size, np.nan)])
    return output[:chains]


def _store_smc_log_marginal_likelihood(
    idata: az.InferenceData,
    trace,
    *,
    chains: int,
) -> None:
    report = getattr(trace, "report", None)
    if report is None or not hasattr(report, "log_marginal_likelihood"):
        return
    values = _parse_log_marginal_likelihood(
        report.log_marginal_likelihood, chains=chains
    )
    if values is None:
        return
    try:
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), values)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = values.tolist()


def smc_log_evidence(idata: az.InferenceData) -> float:
    """Extract the mean finite SMC log marginal likelihood across chains."""

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "log_marginal_likelihood" in getattr(
        sample_stats, "data_vars", {}
    ):
        values = np.asarray(
            sample_stats["log_marginal_likelihood"].values, dtype=float
        )
    else:
        raw = getattr(idata, "attrs", {}).get("log_marginal_likelihood")
        if raw is None:
            raise RuntimeError("SMC log marginal likelihood is missing")
        values = np.asarray(raw, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        raise RuntimeError("SMC log marginal likelihood is not finite")
    return float(values.mean())


def _gamma_shape_from_mean_sd(mu, sd):
    sd = pt.maximum(sd, 1e-12)
    return pt.square(mu / sd)


def build_count_model(
    model_name: str,
    counts_per_cell: Sequence[int],
    exposure: float | Sequence[float],
    *,
    lambda_prior_bounds: tuple[float, float] = (-5.0, 2.0),
    p_prior: tuple[float, float] = (1.0, 1.0),
    std_prior_scale: float = 1.0,
) -> pm.Model:
    """Build one homogeneous, zero-inflated, or Gamma-Poisson count model."""

    if model_name not in MODEL_ORDER:
        raise ValueError(f"unknown model_name {model_name!r}")
    counts, exposure_array = _validate_counts(counts_per_cell, exposure)
    if lambda_prior_bounds[1] <= lambda_prior_bounds[0]:
        raise ValueError("lambda_prior_bounds must be increasing")
    if p_prior[0] <= 0 or p_prior[1] <= 0:
        raise ValueError("p_prior parameters must be positive")
    if std_prior_scale <= 0:
        raise ValueError("std_prior_scale must be positive")

    with pm.Model() as model:
        eta = pm.Uniform(
            "eta",
            lower=float(lambda_prior_bounds[0]),
            upper=float(lambda_prior_bounds[1]),
        )
        if model_name in {"homo", "Z2P"}:
            rate = pm.Deterministic("lambda", 10.0**eta)
            mean_counts = rate * exposure_array
            if model_name == "homo":
                pm.Poisson("counts", mu=mean_counts, observed=counts)
            else:
                p_zero = pm.Beta(
                    "p_zero", alpha=float(p_prior[0]), beta=float(p_prior[1])
                )
                pm.ZeroInflatedPoisson(
                    "counts", psi=1.0 - p_zero, mu=mean_counts, observed=counts
                )
        else:
            mu_lambda = pm.Deterministic("mu_lambda", 10.0**eta)
            sigma_lambda = pm.HalfNormal(
                "sigma_lambda", sigma=float(std_prior_scale)
            )
            alpha = pm.Deterministic(
                "alpha_gamma", _gamma_shape_from_mean_sd(mu_lambda, sigma_lambda)
            )
            mean_counts = mu_lambda * exposure_array
            if model_name == "Dis2P":
                pm.NegativeBinomial(
                    "counts", mu=mean_counts, alpha=alpha, observed=counts
                )
            else:
                p_zero = pm.Beta(
                    "p_zero", alpha=float(p_prior[0]), beta=float(p_prior[1])
                )
                pm.ZeroInflatedNegativeBinomial(
                    "counts",
                    psi=1.0 - p_zero,
                    mu=mean_counts,
                    alpha=alpha,
                    observed=counts,
                )
    return model


def fit_count_model(
    model_name: str,
    counts_per_cell: Sequence[int],
    exposure: float | Sequence[float],
    *,
    observable: str | None = None,
    draws: int = 3000,
    chains: int = 4,
    cores: int | None = 1,
    lambda_prior_bounds: tuple[float, float] = (-5.0, 2.0),
    p_prior: tuple[float, float] = (1.0, 1.0),
    std_prior_scale: float = 1.0,
    random_seed: int | None = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progressbar: bool = True,
    print_summary: bool = False,
) -> FitResult:
    """Fit one count model with Sequential Monte Carlo."""

    model = build_count_model(
        model_name,
        counts_per_cell,
        exposure,
        lambda_prior_bounds=lambda_prior_bounds,
        p_prior=p_prior,
        std_prior_scale=std_prior_scale,
    )
    with model:
        trace = pm.sample_smc(
            draws=int(draws),
            chains=int(chains),
            cores=_resolve_cores(cores),
            random_seed=random_seed,
            progressbar=bool(progressbar),
            return_inferencedata=False,
            threshold=float(threshold),
            correlation_threshold=float(correlation_threshold),
        )
        converter = getattr(pm, "to_inferencedata", None) or getattr(
            pm, "to_inference_data", None
        )
        if converter is None:
            raise RuntimeError("PyMC cannot convert the SMC trace to InferenceData")
        idata = converter(trace, log_likelihood=True)
        _store_smc_log_marginal_likelihood(idata, trace, chains=int(chains))
        if print_summary:
            print(az.summary(idata, hdi_prob=0.95))

    return FitResult(
        model_name=model_name,
        model_label=MODEL_LABELS[model_name],
        model=model,
        idata=idata,
        log_evidence=smc_log_evidence(idata),
        observable=observable,
    )


def fit_model_suite(
    counts_per_cell: Sequence[int],
    exposure: float | Sequence[float],
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    observable: str | None = None,
    **fit_kwargs,
) -> list[FitResult]:
    """Fit all requested count models to the same per-cell observations."""

    return [
        fit_count_model(
            model_name,
            counts_per_cell,
            exposure,
            observable=observable,
            **fit_kwargs,
        )
        for model_name in model_order
    ]


def infer_result(
    result: SimulationResult,
    *,
    observable: Observable = "kills",
    exposure: float | Sequence[float] | None = None,
    **fit_kwargs,
) -> list[FitResult]:
    """Fit the model suite to contacts, synapses, or kills from a simulation."""

    counts = result.counts_per_cell(observable)
    resolved_exposure = result.duration if exposure is None else exposure
    return fit_model_suite(
        counts,
        resolved_exposure,
        observable=observable,
        **fit_kwargs,
    )


def infer_contacts(result: SimulationResult, **fit_kwargs) -> list[FitResult]:
    """Fit the Bayesian model suite to proximity contacts per killer."""

    return infer_result(result, observable="contacts", **fit_kwargs)


def infer_kills(result: SimulationResult, **fit_kwargs) -> list[FitResult]:
    """Fit the Bayesian model suite to primary-attributed kills per killer."""

    return infer_result(result, observable="kills", **fit_kwargs)


def evidence_table(fits: Sequence[FitResult]) -> pd.DataFrame:
    """Summarise log evidence and equal-prior posterior model probabilities."""

    if not fits:
        raise ValueError("fits must not be empty")
    best = max(float(fit.log_evidence) for fit in fits)
    relative = np.asarray(
        [np.exp(float(fit.log_evidence) - best) for fit in fits], dtype=float
    )
    probabilities = relative / relative.sum()
    rows = []
    for fit, probability in zip(fits, probabilities):
        rows.append(
            {
                "model_name": fit.model_name,
                "model_label": fit.model_label,
                "observable": fit.observable,
                "log_evidence": float(fit.log_evidence),
                "delta_log_evidence": float(fit.log_evidence) - best,
                "posterior_model_probability": float(probability),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "log_evidence", ascending=False, ignore_index=True
    )


def bayes_factor_matrix(fits: Sequence[FitResult], *, log10: bool = False) -> pd.DataFrame:
    """Return pairwise Bayes factors, or their base-10 logarithms."""

    names = [fit.model_name for fit in fits]
    evidence = np.asarray([fit.log_evidence for fit in fits], dtype=float)
    log_matrix = evidence[:, None] - evidence[None, :]
    values = (
        log_matrix / np.log(10.0)
        if log10
        else np.exp(np.clip(log_matrix, -745.0, 709.0))
    )
    return pd.DataFrame(values, index=names, columns=names)


def posterior_summary_table(
    fits: Sequence[FitResult],
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Create a tidy table for all scalar posterior variables."""

    rows: list[dict[str, float | str]] = []
    for fit in fits:
        summary = az.summary(fit.idata, hdi_prob=hdi_prob)
        for variable, values in summary.iterrows():
            rows.append(
                {
                    "model_name": fit.model_name,
                    "variable": str(variable),
                    "mean": float(values["mean"]),
                    "sd": float(values["sd"]),
                    "hdi_lower": float(values.filter(like="hdi_").iloc[0]),
                    "hdi_upper": float(values.filter(like="hdi_").iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def _legacy_fit(model_name: str, counts_per_cell, obs_time: float, **kwargs):
    if "p_prior_bounds" in kwargs:
        kwargs["p_prior"] = kwargs.pop("p_prior_bounds")
    if "std_prior_factor" in kwargs:
        kwargs["std_prior_scale"] = kwargs.pop("std_prior_factor")
    kwargs.setdefault("print_summary", True)
    fit = fit_count_model(model_name, counts_per_cell, obs_time, **kwargs)
    return {"idata": fit.idata, "model": fit.model, "fit": fit}


def inference_homo(counts_per_cell, obs_time: float, **kwargs):
    return _legacy_fit("homo", counts_per_cell, obs_time, **kwargs)


def inference_Z2P(counts_per_cell, obs_time: float, **kwargs):
    return _legacy_fit("Z2P", counts_per_cell, obs_time, **kwargs)


def inference_Dis2P(counts_per_cell, obs_time: float, **kwargs):
    return _legacy_fit("Dis2P", counts_per_cell, obs_time, **kwargs)


def inference_hetero3(counts_per_cell, obs_time: float, **kwargs):
    return _legacy_fit("hetero3", counts_per_cell, obs_time, **kwargs)


__all__ = [
    "FitResult",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "bayes_factor_matrix",
    "build_count_model",
    "evidence_table",
    "fit_count_model",
    "fit_model_suite",
    "infer_contacts",
    "infer_kills",
    "infer_result",
    "inference_Dis2P",
    "inference_Z2P",
    "inference_hetero3",
    "inference_homo",
    "posterior_summary_table",
    "smc_log_evidence",
]
