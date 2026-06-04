from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MODEL_ORDER: tuple[str, ...] = ("homo", "Dis2P", "Z2P", "hetero3")
MODEL_LABELS: Mapping[str, str] = {
    "homo": "Homogeneous Poisson",
    "Dis2P": "Distributed Poisson",
    "Z2P": "Zero-Inflated Poisson",
    "hetero3": "Zero-Inflated Distributed Poisson",
}


@dataclass(frozen=True)
class FitResult:
    model_name: str
    model_label: str
    model: pm.Model
    idata: az.InferenceData
    log_evidence: float


def _prepare_arrays(counts: Sequence[int], obs_time: float | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    count_array = np.asarray(counts, dtype=np.int64).reshape(-1)
    if count_array.size == 0:
        raise ValueError("counts must contain at least one observation")
    if np.any(count_array < 0):
        raise ValueError("counts must be non-negative")

    if np.isscalar(obs_time):
        exposure_array = np.full(count_array.shape, float(obs_time), dtype=float)
    else:
        exposure_array = np.asarray(obs_time, dtype=float).reshape(-1)

    if exposure_array.shape != count_array.shape:
        raise ValueError("obs_time must be a scalar or an array aligned with counts")
    if np.any(exposure_array <= 0):
        raise ValueError("exposure must be strictly positive")
    return count_array, exposure_array


def _smc_log_evidence(idata: az.InferenceData) -> float:
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "log_marginal_likelihood" in getattr(sample_stats, "data_vars", {}):
        values = np.asarray(sample_stats["log_marginal_likelihood"].values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            return float(values.mean())

    raw = idata.attrs.get("log_marginal_likelihood")
    if raw is None:
        raise RuntimeError("SMC evidence missing from inference data")
    values = np.asarray(raw, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("SMC evidence present but not finite")
    return float(values.mean())


def smc_log_evidence(idata: az.InferenceData) -> float:
    return _smc_log_evidence(idata)


def _extract_log_marginal_likelihood(trace, *, chains: int) -> np.ndarray:
    report = getattr(trace, "report", None)
    if report is None or not hasattr(report, "log_marginal_likelihood"):
        raise RuntimeError("SMC trace does not expose log_marginal_likelihood")

    raw = report.log_marginal_likelihood
    if isinstance(raw, (list, tuple, np.ndarray)):
        values: list[float] = []
        for item in raw:
            try:
                item_array = np.asarray(item, dtype=float).ravel()
            except Exception:
                continue
            item_array = item_array[np.isfinite(item_array)]
            if item_array.size:
                values.append(float(item_array[-1]))
        logml = np.asarray(values, dtype=float)
    else:
        logml = np.full(int(chains), float(raw), dtype=float)

    if logml.size == 0:
        raise RuntimeError("No valid SMC marginal likelihood values found")
    if logml.size < int(chains):
        padding = np.full(int(chains) - logml.size, np.nan, dtype=float)
        logml = np.concatenate([logml, padding])
    return logml[: int(chains)]


def _zero_inflated_logp(base_logp, counts_t, p_zero):
    return pt.switch(
        pt.eq(counts_t, 0),
        pt.logaddexp(pt.log(p_zero), pt.log1p(-p_zero) + base_logp),
        pt.log1p(-p_zero) + base_logp,
    )


def build_model(
    model_name: str,
    counts: Sequence[int],
    obs_time: float | Sequence[float],
    *,
    dis_mode: str = "gamma",
) -> pm.Model:
    count_array, exposure_array = _prepare_arrays(counts, obs_time)
    counts_t = pt.as_tensor_variable(count_array.astype("int64"))
    exposure_t = pt.as_tensor_variable(exposure_array.astype("float64"))

    with pm.Model() as model:
        if model_name == "homo":
            lambda_rate = pm.Gamma("lambda", alpha=2.0, beta=2.0)
            base_logp = pm.logp(pm.Poisson.dist(mu=lambda_rate * exposure_t), counts_t)
            pm.Potential("likelihood", pt.sum(base_logp))

        elif model_name == "Z2P":
            lambda_rate = pm.Gamma("lambda", alpha=2.0, beta=2.0)
            p_zero = pm.Beta("p_zero", alpha=1.0, beta=1.0)
            base_logp = pm.logp(pm.Poisson.dist(mu=lambda_rate * exposure_t), counts_t)
            pm.Potential("likelihood", pt.sum(_zero_inflated_logp(base_logp, counts_t, p_zero)))

        elif model_name in {"Dis2P", "hetero3"}:
            if dis_mode != "gamma":
                raise ValueError("Only dis_mode='gamma' is currently implemented")

            mu_lambda = pm.Gamma("mu_lambda", alpha=2.0, beta=2.0)
            sigma_lambda = pm.Exponential("sigma_lambda", lam=1.0)
            alpha_nb = pm.Deterministic("alpha_nb", pt.square(mu_lambda / (sigma_lambda + 1e-9)))
            base_logp = pm.logp(
                pm.NegativeBinomial.dist(mu=mu_lambda * exposure_t, alpha=alpha_nb),
                counts_t,
            )

            if model_name == "hetero3":
                p_zero = pm.Beta("p_zero", alpha=1.0, beta=1.0)
                base_logp = _zero_inflated_logp(base_logp, counts_t, p_zero)

            pm.Potential("likelihood", pt.sum(base_logp))

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    return model


def fit_model(
    model_name: str,
    counts: Sequence[int],
    obs_time: float | Sequence[float],
    *,
    dis_mode: str = "gamma",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores: int | None = 1,
    random_seed: int | None = None,
) -> FitResult:
    del tune, target_accept

    model = build_model(model_name, counts, obs_time, dis_mode=dis_mode)
    with model:
        trace = pm.sample_smc(
            draws=int(draws),
            chains=int(chains),
            cores=None if cores is None else int(cores),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=False,
        )
        to_idata = getattr(pm, "to_inferencedata", None) or getattr(pm, "to_inference_data", None)
        if to_idata is None:
            raise RuntimeError("PyMC does not expose to_inferencedata/to_inference_data")
        idata = to_idata(trace)
        print(az.summary(idata))

    logml = _extract_log_marginal_likelihood(trace, chains=int(chains))
    try:
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), logml)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = logml.tolist()

    return FitResult(
        model_name=model_name,
        model_label=MODEL_LABELS[model_name],
        model=model,
        idata=idata,
        log_evidence=_smc_log_evidence(idata),
    )


def fit_model_suite(
    counts: Sequence[int],
    obs_time: float | Sequence[float],
    *,
    mode: str = "counts",
    dis_mode: str = "gamma",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores: int | None = 1,
    random_seed: int | None = None,
    model_order: Sequence[str] = MODEL_ORDER,
) -> list[FitResult]:
    if mode != "counts":
        raise ValueError("Only mode='counts' is supported")

    return [
        fit_model(
            model_name,
            counts,
            obs_time,
            dis_mode=dis_mode,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
        )
        for model_name in model_order
    ]


def _posterior_samples(idata: az.InferenceData, var_name: str) -> np.ndarray:
    values = np.asarray(idata.posterior[var_name].values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite posterior draws found for {var_name}")
    return values


def _summarise_posterior(idata: az.InferenceData, var_name: str, *, hdi_prob: float) -> dict[str, float]:
    draws = _posterior_samples(idata, var_name)
    hdi = az.hdi(draws, hdi_prob=hdi_prob)
    return {
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "hdi_lower": float(hdi[0]),
        "hdi_upper": float(hdi[1]),
    }


def evidence_table(fits: Sequence[FitResult]) -> pd.DataFrame:
    best = max(float(fit.log_evidence) for fit in fits)
    weights_raw = {fit.model_name: np.exp(float(fit.log_evidence) - best) for fit in fits}
    weight_norm = float(sum(weights_raw.values()))

    rows = [
        {
            "model_name": fit.model_name,
            "model_label": fit.model_label,
            "log_evidence": float(fit.log_evidence),
            "delta_log_evidence": float(fit.log_evidence) - best,
            "posterior_model_prob_equal_priors": weights_raw[fit.model_name] / weight_norm,
        }
        for fit in fits
    ]
    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(out["model_name"], categories=list(MODEL_ORDER), ordered=True)
    return out.sort_values(["log_evidence", "model_name"], ascending=[False, True]).reset_index(drop=True)


def bayes_factor_matrix(fits: Sequence[FitResult]) -> pd.DataFrame:
    fit_map = {fit.model_name: fit for fit in fits}
    names = [name for name in MODEL_ORDER if name in fit_map]
    matrix_rows: list[dict[str, object]] = []
    for row_name in names:
        row = {"model_name": row_name, "model_label": MODEL_LABELS[row_name]}
        for col_name in names:
            row[col_name] = float(fit_map[row_name].log_evidence) - float(fit_map[col_name].log_evidence)
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows)


def posterior_summary_table(
    fits: Sequence[FitResult],
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fit_map = {fit.model_name: fit for fit in fits}

    for model_name in MODEL_ORDER:
        if model_name not in fit_map:
            continue

        fit = fit_map[model_name]
        row: dict[str, object] = {
            "model_name": fit.model_name,
            "model_label": fit.model_label,
            "lambda_mean": np.nan,
            "lambda_median": np.nan,
            "lambda_sd": np.nan,
            "lambda_hdi_lower": np.nan,
            "lambda_hdi_upper": np.nan,
            "log_evidence": float(fit.log_evidence),
            "p_zero_mean": np.nan,
            "p_zero_median": np.nan,
            "p_zero_sd": np.nan,
            "p_zero_hdi_lower": np.nan,
            "p_zero_hdi_upper": np.nan,
            "mu_lambda_mean": np.nan,
            "mu_lambda_median": np.nan,
            "mu_lambda_sd": np.nan,
            "mu_lambda_hdi_lower": np.nan,
            "mu_lambda_hdi_upper": np.nan,
            "sigma_lambda_mean": np.nan,
            "sigma_lambda_median": np.nan,
            "sigma_lambda_sd": np.nan,
            "sigma_lambda_hdi_lower": np.nan,
            "sigma_lambda_hdi_upper": np.nan,
        }

        if "lambda" in fit.idata.posterior:
            stats = _summarise_posterior(fit.idata, "lambda", hdi_prob=hdi_prob)
            row.update(
                {
                    "lambda_mean": stats["mean"],
                    "lambda_median": stats["median"],
                    "lambda_sd": stats["sd"],
                    "lambda_hdi_lower": stats["hdi_lower"],
                    "lambda_hdi_upper": stats["hdi_upper"],
                }
            )

        if "p_zero" in fit.idata.posterior:
            stats = _summarise_posterior(fit.idata, "p_zero", hdi_prob=hdi_prob)
            row.update(
                {
                    "p_zero_mean": stats["mean"],
                    "p_zero_median": stats["median"],
                    "p_zero_sd": stats["sd"],
                    "p_zero_hdi_lower": stats["hdi_lower"],
                    "p_zero_hdi_upper": stats["hdi_upper"],
                }
            )

        if "mu_lambda" in fit.idata.posterior:
            stats = _summarise_posterior(fit.idata, "mu_lambda", hdi_prob=hdi_prob)
            row.update(
                {
                    "mu_lambda_mean": stats["mean"],
                    "mu_lambda_median": stats["median"],
                    "mu_lambda_sd": stats["sd"],
                    "mu_lambda_hdi_lower": stats["hdi_lower"],
                    "mu_lambda_hdi_upper": stats["hdi_upper"],
                }
            )

        if "sigma_lambda" in fit.idata.posterior:
            stats = _summarise_posterior(fit.idata, "sigma_lambda", hdi_prob=hdi_prob)
            row.update(
                {
                    "sigma_lambda_mean": stats["mean"],
                    "sigma_lambda_median": stats["median"],
                    "sigma_lambda_sd": stats["sd"],
                    "sigma_lambda_hdi_lower": stats["hdi_lower"],
                    "sigma_lambda_hdi_upper": stats["hdi_upper"],
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    out["model_name"] = pd.Categorical(out["model_name"], categories=list(MODEL_ORDER), ordered=True)
    return out.sort_values(["log_evidence", "model_name"], ascending=[False, True]).reset_index(drop=True)
