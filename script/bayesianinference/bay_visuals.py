# visua.py
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pathlib import Path
import pymc as pm
import arviz as az


FONT_SIZE = 20
LABEL_SIZE = 24
LEGEND_SIZE = 20
TICK_SIZE = 20

plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans Mono", "monospace"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
    }
)

Edge = Tuple[Tuple[int, int], Tuple[int, int]]


def node_counts_from_history(history: np.ndarray) -> Dict[Tuple[int, int], int]:
    if history.ndim != 2 or history.shape[1] != 4:
        raise ValueError("history must have shape (n_cells, 4)")
    n_cells = int(history.shape[0])
    node_counts: Dict[Tuple[int, int], int] = {(0, 0): n_cells}
    cum_success = np.cumsum(history.astype(np.int8), axis=1)
    for i in range(1, 5):
        succ_i = cum_success[:, i - 1]
        for j in range(i + 1):
            node_counts[(i - j, j)] = int(np.sum(succ_i == j))
    return node_counts


def edge_counts_from_history(history: np.ndarray) -> Dict[Edge, int]:
    if history.ndim != 2 or history.shape[1] != 4:
        raise ValueError("history must have shape (n_cells, 4)")
    n_cells = int(history.shape[0])
    h = history.astype(np.int8)
    cum_success = np.cumsum(h, axis=1)

    edges: Dict[Edge, int] = {}
    for t in range(4):
        succ_prev = np.zeros(n_cells, dtype=np.int8) if t == 0 else cum_success[:, t - 1]
        y = h[:, t]

        for j in range(t + 1):
            mask = succ_prev == j
            if not np.any(mask):
                continue
            f = int(t - j)
            s = int(j)

            n1 = int(np.sum(y[mask] == 1))
            n0 = int(mask.sum() - n1)

            if n0:
                e0 = ((f, s), (f + 1, s))
                edges[e0] = edges.get(e0, 0) + n0
            if n1:
                e1 = ((f, s), (f, s + 1))
                edges[e1] = edges.get(e1, 0) + n1

    return edges


# def plot_pascal(
#     layers: int,
#     *,
#     n_cells: int,
#     node_values: Dict[Tuple[int, int], float] | None = None,
#     node_counts: Dict[Tuple[int, int], int] | None = None,
#     history: np.ndarray | None = None,
#     step=2,
#     box_size=1.0,
#     figsize=None,
#     lw=1.4,
#     fontsize=10,
#     margin=2,
#     arrowstyle="->",
#     cmap_name="coolwarm",
#     vmin=0.0,
#     vmax=1.0,
#     cmap_min: float = 0.5,
#     cmap_max: float = 0.8,
#     label_mode: str = "p",
#     arrow_lw_min: float = 0.6,
#     arrow_lw_max: float = 6.0,
#     arrow_label_fmt: str = "{p:.3f}",
#     show_arrow_labels: bool = True,
#     arrow_label_color: str = "royalblue",
#     arrow_label_offset: float = 0.55,
#     arrow_label_alpha: float = 0.75,
#     arrow_label_pos: float = 0.40,
#     title: str | None = None,
# ):
#     if layers < 1:
#         raise ValueError("layers must be >= 1")
#     if step < 1:
#         raise ValueError("step must be >= 1")
#     if box_size <= 0:
#         raise ValueError("box_size must be > 0")
#     if box_size >= step:
#         raise ValueError("box_size must be < step to avoid touching")
#     if not (0.0 <= cmap_min < cmap_max <= 1.0):
#         raise ValueError("cmap_min/cmap_max must satisfy 0 <= cmap_min < cmap_max <= 1")

#     allowed_label_modes = {"both", "n", "p", "none"}
#     if label_mode not in allowed_label_modes:
#         raise ValueError(f"label_mode must be one of {sorted(allowed_label_modes)}")

#     if history is not None:
#         if node_counts is None:
#             node_counts = node_counts_from_history(history)
#         edge_counts = edge_counts_from_history(history)
#     else:
#         edge_counts = None

#     if figsize is None:
#         figsize = (max(6, 1.6 * layers), max(5, 1.4 * layers))
#     fig, ax = plt.subplots(figsize=figsize)

#     half = box_size / 2

#     base_cmap = plt.get_cmap(cmap_name)
#     if cmap_min == 0.0 and cmap_max == 1.0:
#         cmap = base_cmap
#     else:
#         from matplotlib.colors import LinearSegmentedColormap

#         cmap = LinearSegmentedColormap.from_list(
#             f"{base_cmap.name}_trunc_{cmap_min:.2f}_{cmap_max:.2f}",
#             base_cmap(np.linspace(cmap_min, cmap_max, 256)),
#         )
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     def node_xy(fail: int, succ: int):
#         i = fail + succ
#         x = step * (succ - fail)
#         y = -step * i
#         return float(x), float(y)

#     def clip_to_box_boundary(src, dst):
#         sx, sy = src
#         tx, ty = dst
#         dx, dy = tx - sx, ty - sy
#         if dx == 0 and dy == 0:
#             return src, dst
#         ax_abs = abs(dx) / half if dx != 0 else float("inf")
#         ay_abs = abs(dy) / half if dy != 0 else float("inf")
#         t = 1.0 / max(ax_abs, ay_abs)
#         p0 = (sx + dx * t, sy + dy * t)
#         p1 = (tx - dx * t, ty - dy * t)
#         return p0, p1

#     def place_edge_label(p0, p1, txt):
#         x0, y0 = p0
#         x1, y1 = p1
#         dx, dy = (x1 - x0), (y1 - y0)
#         L = float(np.hypot(dx, dy))
#         if L == 0.0:
#             return
#         nx, ny = (-dy / L), (dx / L)
#         xm = x0 + float(arrow_label_pos) * dx
#         ym = y0 + float(arrow_label_pos) * dy
#         ax.text(
#             xm + arrow_label_offset * nx,
#             ym + arrow_label_offset * ny,
#             txt,
#             ha="center",
#             va="center",
#             fontsize=max(8, fontsize - 2),
#             color=arrow_label_color,
#             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=arrow_label_alpha),
#         )

#     nodes = {(f, s): node_xy(f, s) for i in range(layers) for s in range(i + 1) for f in [i - s]}

#     for (f, s), (x, y) in nodes.items():
#         val = None if node_values is None else node_values.get((f, s), 0.0)
#         facecolor = "white" if val is None else cmap(norm(val))
#         ax.add_patch(
#             Rectangle(
#                 (x - half, y - half),
#                 box_size,
#                 box_size,
#                 fill=True,
#                 facecolor=facecolor,
#                 edgecolor="black",
#                 linewidth=lw,
#             )
#         )
#         if label_mode != "none":
#             cnt = None if node_counts is None else node_counts.get((f, s), None)
#             parts = [f"({f},{s})"]
#             if label_mode in {"both", "n"} and cnt is not None:
#                 parts.append(f"{cnt}")
#             if label_mode in {"both", "p"} and val is not None:
#                 parts.append(f"{val:.3f}")
#             ax.text(x, y, "\n".join(parts), ha="center", va="center", fontsize=fontsize)

#     def edge_lw(count: int) -> float:
#         if edge_counts is None:
#             return lw
#         p = count / float(n_cells)
#         return arrow_lw_min + (arrow_lw_max - arrow_lw_min) * float(np.clip(p, 0.0, 1.0))

#     def edge_label(count: int) -> str:
#         p = count / float(n_cells)
#         return arrow_label_fmt.format(p=p)

#     for i in range(layers - 1):
#         for succ in range(i + 1):
#             fail = i - succ
#             a = nodes[(fail, succ)]
#             b_fail = nodes[(fail + 1, succ)]
#             b_succ = nodes[(fail, succ + 1)]

#             cnt0 = None
#             cnt1 = None
#             if edge_counts is not None:
#                 cnt0 = edge_counts.get(((fail, succ), (fail + 1, succ)), 0)
#                 cnt1 = edge_counts.get(((fail, succ), (fail, succ + 1)), 0)

#             s0, e0 = clip_to_box_boundary(a, b_fail)
#             lw0 = edge_lw(cnt0) if cnt0 is not None else lw
#             ax.annotate("", xy=e0, xytext=s0, arrowprops=dict(arrowstyle=arrowstyle, lw=lw0))
#             if show_arrow_labels and cnt0 is not None and cnt0 > 0:
#                 place_edge_label(s0, e0, edge_label(cnt0))

#             s1, e1 = clip_to_box_boundary(a, b_succ)
#             lw1 = edge_lw(cnt1) if cnt1 is not None else lw
#             ax.annotate("", xy=e1, xytext=s1, arrowprops=dict(arrowstyle=arrowstyle, lw=lw1))
#             if show_arrow_labels and cnt1 is not None and cnt1 > 0:
#                 place_edge_label(s1, e1, edge_label(cnt1))

#     xs = [x for x, _ in nodes.values()]
#     ys = [y for _, y in nodes.values()]
#     ax.set_xlim(min(xs) - margin, max(xs) + margin)
#     ax.set_ylim(min(ys) - margin, max(ys) + margin)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     ax.text(
#         0.98,
#         0.98,
#         f"cell number = {n_cells}",
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=LEGEND_SIZE,
#     )

#     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
#     cbar.set_label("Cell proportion in each state", fontsize=LEGEND_SIZE)

#     if title is not None:
#         ax.set_title(title)

#     return fig, ax

def plot_pascal(
    layers: int,
    *,
    n_cells: int,
    node_values: Dict[Tuple[int, int], float] | None = None,
    node_counts: Dict[Tuple[int, int], int] | None = None,
    history: np.ndarray | None = None,
    step=2,
    box_size=1.0,
    figsize=None,
    lw=1.4,
    fontsize=10,
    margin=2,
    arrowstyle="->",
    cmap_name="coolwarm",
    vmin=0.0,
    vmax=1.0,
    cmap_min: float = 0.5,
    cmap_max: float = 0.8,
    label_mode: str = "p",
    arrow_mode: str = "p_stimul",  # "p_stimul" or "local"
    arrow_lw_min: float = 0.6,
    arrow_lw_max: float = 6.0,
    arrow_label_fmt: str = "{p:.3f}",
    show_arrow_labels: bool = True,
    arrow_label_color: str = "royalblue",
    arrow_label_offset: float = 0.55,
    arrow_label_alpha: float = 0.75,
    arrow_label_pos: float = 0.40,
    title: str | None = None,
):
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")
    if box_size >= step:
        raise ValueError("box_size must be < step to avoid touching")
    if not (0.0 <= cmap_min < cmap_max <= 1.0):
        raise ValueError("cmap_min/cmap_max must satisfy 0 <= cmap_min < cmap_max <= 1")

    allowed_label_modes = {"both", "n", "p", "none"}
    if label_mode not in allowed_label_modes:
        raise ValueError(f"label_mode must be one of {sorted(allowed_label_modes)}")

    allowed_arrow_modes = {"local", "p_stimul"}
    if arrow_mode not in allowed_arrow_modes:
        raise ValueError(f"arrow_mode must be one of {sorted(allowed_arrow_modes)}")

    if history is not None:
        if node_counts is None:
            node_counts = node_counts_from_history(history)
        edge_counts = edge_counts_from_history(history)

        if node_values is None:
            node_values = {k: v / float(n_cells) for k, v in node_counts.items()}
    else:
        edge_counts = None

    if figsize is None:
        figsize = (max(6, 1.6 * layers), max(5, 1.4 * layers))
    fig, ax = plt.subplots(figsize=figsize)

    half = box_size / 2

    base_cmap = plt.get_cmap(cmap_name)
    if cmap_min == 0.0 and cmap_max == 1.0:
        cmap = base_cmap
    else:
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(
            f"{base_cmap.name}_trunc_{cmap_min:.2f}_{cmap_max:.2f}",
            base_cmap(np.linspace(cmap_min, cmap_max, 256)),
        )
    norm = Normalize(vmin=vmin, vmax=vmax)

    def node_xy(fail: int, succ: int):
        i = fail + succ
        x = step * (succ - fail)
        y = -step * i
        return float(x), float(y)

    def clip_to_box_boundary(src, dst):
        sx, sy = src
        tx, ty = dst
        dx, dy = tx - sx, ty - sy
        if dx == 0 and dy == 0:
            return src, dst
        ax_abs = abs(dx) / half if dx != 0 else float("inf")
        ay_abs = abs(dy) / half if dy != 0 else float("inf")
        t = 1.0 / max(ax_abs, ay_abs)
        p0 = (sx + dx * t, sy + dy * t)
        p1 = (tx - dx * t, ty - dy * t)
        return p0, p1

    def place_edge_label(p0, p1, txt):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = (x1 - x0), (y1 - y0)
        L = float(np.hypot(dx, dy))
        if L == 0.0:
            return
        nx, ny = (-dy / L), (dx / L)
        xm = x0 + float(arrow_label_pos) * dx
        ym = y0 + float(arrow_label_pos) * dy
        ax.text(
            xm + arrow_label_offset * nx,
            ym + arrow_label_offset * ny,
            txt,
            ha="center",
            va="center",
            fontsize=max(8, fontsize - 2),
            color=arrow_label_color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=arrow_label_alpha),
        )

    nodes = {(f, s): node_xy(f, s) for i in range(layers) for s in range(i + 1) for f in [i - s]}

    for (f, s), (x, y) in nodes.items():
        val = None if node_values is None else node_values.get((f, s), 0.0)
        facecolor = "white" if val is None else cmap(norm(val))
        ax.add_patch(
            Rectangle(
                (x - half, y - half),
                box_size,
                box_size,
                fill=True,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=lw,
            )
        )
        if label_mode != "none":
            cnt = None if node_counts is None else node_counts.get((f, s), None)
            parts = [f"({f},{s})"]
            if label_mode in {"both", "n"} and cnt is not None:
                parts.append(f"{cnt}")
            if label_mode in {"both", "p"} and val is not None:
                parts.append(f"{val:.3f}")
            ax.text(x, y, "\n".join(parts), ha="center", va="center", fontsize=fontsize)

    layer_denom: list[float] = [float(n_cells)] * max(1, layers - 1)
    if edge_counts is not None and arrow_mode == "p_stimul":
        for i in range(layers - 1):
            tot = 0
            for succ in range(i + 1):
                fail = i - succ
                parent = (fail, succ)
                tot += int(edge_counts.get((parent, (fail + 1, succ)), 0))
                tot += int(edge_counts.get((parent, (fail, succ + 1)), 0))
            layer_denom[i] = float(tot) if tot > 0 else float(n_cells)

    def edge_prob(i_layer: int, parent: Tuple[int, int], cnt: int, sibling_cnt: int) -> float:
        if edge_counts is None:
            return 0.0
        if arrow_mode == "local":
            denom = float(node_counts.get(parent, 0)) if node_counts is not None else float(cnt + sibling_cnt)
        else:
            denom = float(layer_denom[i_layer])
        if denom <= 0.0:
            return 0.0
        return float(cnt) / denom

    def edge_lw(p: float) -> float:
        return arrow_lw_min + (arrow_lw_max - arrow_lw_min) * float(np.clip(p, 0.0, 1.0))

    for i in range(layers - 1):
        for succ in range(i + 1):
            fail = i - succ
            parent = (fail, succ)

            a = nodes[parent]
            b_fail = nodes[(fail + 1, succ)]
            b_succ = nodes[(fail, succ + 1)]

            cnt0 = cnt1 = 0
            if edge_counts is not None:
                cnt0 = int(edge_counts.get((parent, (fail + 1, succ)), 0))
                cnt1 = int(edge_counts.get((parent, (fail, succ + 1)), 0))

            p0 = edge_prob(i, parent, cnt0, cnt1)
            p1 = edge_prob(i, parent, cnt1, cnt0)

            s0, e0 = clip_to_box_boundary(a, b_fail)
            ax.annotate("", xy=e0, xytext=s0, arrowprops=dict(arrowstyle=arrowstyle, lw=edge_lw(p0) if edge_counts is not None else lw))
            if show_arrow_labels and edge_counts is not None and (cnt0 > 0 or cnt1 > 0):
                place_edge_label(s0, e0, arrow_label_fmt.format(p=p0))

            s1, e1 = clip_to_box_boundary(a, b_succ)
            ax.annotate("", xy=e1, xytext=s1, arrowprops=dict(arrowstyle=arrowstyle, lw=edge_lw(p1) if edge_counts is not None else lw))
            if show_arrow_labels and edge_counts is not None and (cnt0 > 0 or cnt1 > 0):
                place_edge_label(s1, e1, arrow_label_fmt.format(p=p1))

    xs = [x for x, _ in nodes.values()]
    ys = [y for _, y in nodes.values()]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.98,
        0.98,
        f"cell number = {n_cells}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=LEGEND_SIZE,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Cell proportion in each state", fontsize=LEGEND_SIZE)

    if title is not None:
        ax.set_title(title)

    return fig, ax




def plot_posterior(
    *,
    idata,
    out_path,
    title: str,
    # New versatile API
    params: Sequence[str] | None = None,
    param_display: Mapping[str, str] | None = None,
    ground_truth: Mapping[str, float] | None = None,
    # New: manual axis limits per parameter
    xlim: Mapping[str, tuple[float, float]] | None = None,
    # Convenience API (single panel)
    var_names: Sequence[str] | None = None,
    truth: Mapping[str, float | None] | None = None,
    # Legacy API (4 panels: one per stimulus)
    keys: Sequence[str] | None = None,
    truth_mu: Sequence[float] | None = None,
    truth_sigma: Sequence[float] | None = None,
    truth_phi0: Sequence[float] | None = None,
    dpi: int = 350,
    sample_limit: int = 6000,
    seed: int = 0,
    color: str = "black",
    truth_color: str = "blue",
    hdi_prob: float = 0.95,
) -> None:
    from pathlib import Path

    import arviz as az
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator

    posterior = idata.posterior
    xlim = dict(xlim or {})

    alias_map = {
        "phi0": ("phi0", "phi", "p_zero", "phi_zero"),
        "phi_zero": ("phi_zero", "phi0", "phi", "p_zero"),
        "p_zero": ("p_zero", "phi0", "phi", "phi_zero"),
        "mu": ("mu", "mu_lambda"),
        "mu_lambda": ("mu_lambda", "mu"),
        "sigma": ("sigma", "sigma_lambda"),
        "sigma_lambda": ("sigma_lambda", "sigma"),
        "lambda": ("lambda", "lam"),
        "lam": ("lam", "lambda"),
    }

    def _resolve_varname(varname: str) -> str:
        cands = alias_map.get(varname, (varname,))
        for v in cands:
            if v in posterior:
                return v
        raise KeyError(f"'{varname}' not in idata.posterior")

    def _pull_1d(varname: str, *, idx: int | None = None) -> np.ndarray:
        vn = _resolve_varname(varname)
        x = posterior[vn].stack(sample=("chain", "draw")).values
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            vals = x
        else:
            vals = x.ravel() if idx is None else x[idx].ravel()

        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return vals

        if sample_limit is not None and vals.size > int(sample_limit):
            idx_i = 0 if idx is None else int(idx)
            rng = np.random.default_rng(seed + 13 * idx_i + (7 if varname == "sigma" else 0))
            sel = rng.choice(vals.size, int(sample_limit), replace=False)
            vals = vals[sel]
        return vals

    def _df_for_params(pars: Sequence[str], *, idx: int | None = None) -> pd.DataFrame:
        data = {p: _pull_1d(p, idx=idx) for p in pars}
        lengths = [len(v) for v in data.values() if len(v) > 0]
        if not lengths:
            return pd.DataFrame()
        n = min(lengths)
        data = {k: v[:n] for k, v in data.items()}
        return pd.DataFrame(data)

    def _auto_xlim(vals: np.ndarray, gt: float, *, lower=None, upper=None) -> tuple[float, float]:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            lo, hi = (gt - 1.0, gt + 1.0)
        else:
            qlo, qhi = np.quantile(v, [0.01, 0.99])
            lo = float(min(qlo, gt))
            hi = float(max(qhi, gt))
            pad = 0.15 * max(hi - lo, 1e-9)
            lo -= pad
            hi += pad
        if lower is not None:
            lo = max(lo, float(lower))
        if upper is not None:
            hi = min(hi, float(upper))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        return float(lo), float(hi)

    def _resolve_xlim_for_param(pname: str, vals: np.ndarray, gt_val: float) -> tuple[float, float]:
        if pname in xlim:
            lo, hi = xlim[pname]
            return float(lo), float(hi)

        if pname in {"phi0", "phi_zero", "p_zero", "phi"}:
            return _auto_xlim(vals, gt_val, lower=0.0, upper=1.0)
        if pname in {"mu", "mu_lambda", "sigma", "sigma_lambda", "lambda", "lam"}:
            return _auto_xlim(vals, gt_val, lower=0.0)
        if pname in {"p", "mean_p", "p_active"}:
            return _auto_xlim(vals, gt_val, lower=0.0, upper=1.0)
        if pname in {"sd_p"}:
            return _auto_xlim(vals, gt_val, lower=0.0)
        return _auto_xlim(vals, gt_val)

    def _plot_triangle(ax_grid, df: pd.DataFrame, pars: Sequence[str], gt: dict, xlims: dict, display: dict) -> None:
        for irow, rowpar in enumerate(pars):
            for icol, colpar in enumerate(pars):
                ax = ax_grid[irow, icol]
                ax.set_facecolor("none")

                if icol > irow:
                    ax.axis("off")
                    continue

                if icol == irow:
                    vals = df[rowpar].dropna().to_numpy()
                    if vals.size == 0:
                        ax.axis("off")
                        continue

                    sns.histplot(
                        vals,
                        bins=30,
                        stat="density",
                        kde=False,
                        ax=ax,
                        color=color,
                        alpha=0.18,
                        element="step",
                        fill=True,
                    )
                    sns.histplot(
                        vals,
                        bins=30,
                        stat="density",
                        kde=False,
                        ax=ax,
                        color=color,
                        alpha=1.0,
                        element="step",
                        fill=False,
                        linewidth=1.8,
                    )

                    try:
                        lo_h, hi_h = az.hdi(vals, hdi_prob=hdi_prob)
                        ax.axvspan(float(lo_h), float(hi_h), color=color, alpha=0.10, linewidth=0)
                    except Exception:
                        pass

                    if rowpar in gt and np.isfinite(float(gt[rowpar])):
                        ax.axvline(float(gt[rowpar]), color=truth_color, linestyle="-", linewidth=1.8)

                    ax.set_xlabel(display.get(rowpar, rowpar), fontsize=LABEL_SIZE)
                    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
                    ax.grid(alpha=0.2)
                    ax.set_xlim(*xlims[rowpar])

                else:
                    if df[colpar].dropna().empty or df[rowpar].dropna().empty:
                        ax.axis("off")
                        continue

                    sns.kdeplot(
                        x=df[colpar],
                        y=df[rowpar],
                        ax=ax,
                        fill=False,
                        color=color,
                        alpha=0.6,
                        levels=7,
                        linewidths=1.0,
                    )
                    if (
                        colpar in gt
                        and rowpar in gt
                        and np.isfinite(float(gt[colpar]))
                        and np.isfinite(float(gt[rowpar]))
                    ):
                        ax.scatter(
                            float(gt[colpar]),
                            float(gt[rowpar]),
                            marker="*",
                            color=truth_color,
                            s=110,
                            linewidths=2.0,
                            zorder=1000,
                        )

                    ax.set_xlabel(display.get(colpar, colpar), fontsize=LABEL_SIZE)
                    ax.set_ylabel(display.get(rowpar, rowpar), fontsize=LABEL_SIZE)
                    ax.grid(alpha=0.25)
                    ax.set_xlim(*xlims[colpar])
                    ax.set_ylim(*xlims[rowpar])

                ax.margins(0.05)
                ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    default_display = {
        "mu": r"$\mu_{\lambda}$",
        "mu_lambda": r"$\mu_{\lambda}$",
        "sigma": r"$\sigma_{\lambda}$",
        "sigma_lambda": r"$\sigma_{\lambda}$",
        "phi0": r"$\phi_0$",
        "phi_zero": r"$\phi_0$",
        "p_zero": r"$\phi_0$",
        "phi": r"$\phi_0$",
        "lambda": r"$\lambda$",
        "lam": r"$\lambda$",
        "p": r"$p$",
        "mean_p": r"$m$",
        "sd_p": r"$s$",
        "p_active": r"$p_{\mathrm{active}}$",
    }

    # ---- Legacy 4-panel mode -------------------------------------------------
    if keys is not None and truth_mu is not None and truth_sigma is not None and truth_phi0 is not None:
        if len(keys) != 4:
            raise ValueError("keys must have length 4")
        truth_mu = np.asarray(truth_mu, dtype=float)
        truth_sigma = np.asarray(truth_sigma, dtype=float)
        truth_phi0 = np.asarray(truth_phi0, dtype=float)
        if truth_mu.shape != (4,) or truth_sigma.shape != (4,) or truth_phi0.shape != (4,):
            raise ValueError("truth arrays must each have shape (4,)")

        pars = ["mu", "sigma", "phi0"]

        display = dict(default_display)
        if param_display is not None:
            display.update(dict(param_display))

        fig = plt.figure(figsize=(22, 18), dpi=dpi)
        fig.patch.set_alpha(0.0)
        outer = gridspec.GridSpec(2, 2, wspace=0.22, hspace=0.22)

        for i in range(4):
            df = _df_for_params(pars, idx=i)
            gt = {"mu": float(truth_mu[i]), "sigma": float(truth_sigma[i]), "phi0": float(truth_phi0[i])}

            xlims = {}
            for p in pars:
                vals = df[p].to_numpy()
                xlims[p] = _resolve_xlim_for_param(p, vals, gt[p])

            sub = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[i], wspace=0.55, hspace=0.55)
            ax_grid = np.empty((3, 3), dtype=object)
            for r in range(3):
                for c in range(3):
                    ax_grid[r, c] = fig.add_subplot(sub[r, c])

            _plot_triangle(ax_grid, df, pars, gt, xlims, display)

            anchor = fig.add_subplot(outer[i])
            anchor.axis("off")
            anchor.text(
                0.5,
                1.03,
                f"{keys[i]}",
                ha="center",
                va="bottom",
                fontsize=LABEL_SIZE,
                transform=anchor.transAxes,
            )

        fig.suptitle(title, fontsize=LABEL_SIZE, y=0.99)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Saved: {out_path}")
        return

    # ---- Versatile single-panel mode ----------------------------------------
    pars = list(params) if params is not None else list(var_names) if var_names is not None else None
    if pars is None:
        pars = ["mu", "sigma", "phi0"]

    if len(pars) < 1:
        raise ValueError("Need at least one parameter to plot")

    gt_map = dict(ground_truth or {})
    if truth is not None:
        for k, v in truth.items():
            if v is not None:
                gt_map[str(k)] = float(v)

    if "phi" in gt_map and "phi0" not in gt_map:
        gt_map["phi0"] = float(gt_map["phi"])
    if "phi_zero" in gt_map and "phi0" not in gt_map:
        gt_map["phi0"] = float(gt_map["phi_zero"])
    if "p_zero" in gt_map and "phi0" not in gt_map:
        gt_map["phi0"] = float(gt_map["p_zero"])
    if "lam" in gt_map and "lambda" not in gt_map:
        gt_map["lambda"] = float(gt_map["lam"])

    df = _df_for_params(pars, idx=None)
    if df.empty:
        raise ValueError("No plottable posterior samples found for requested params")

    display = dict(default_display)
    if param_display is not None:
        display.update(dict(param_display))

    xlims = {}
    for p in pars:
        gt_val = float(gt_map[p]) if p in gt_map else float(np.nanmedian(df[p].to_numpy()))
        xlims[p] = _resolve_xlim_for_param(p, df[p].to_numpy(), gt_val)

    n = len(pars)
    fig = plt.figure(figsize=(6 * n, 6 * n), dpi=dpi)
    fig.patch.set_alpha(0.0)
    gs = gridspec.GridSpec(n, n, wspace=0.5, hspace=0.5)
    ax_grid = np.empty((n, n), dtype=object)
    for r in range(n):
        for c in range(n):
            ax_grid[r, c] = fig.add_subplot(gs[r, c])

    _plot_triangle(ax_grid, df, pars, gt_map, xlims, display)
    fig.suptitle(title, fontsize=LABEL_SIZE, y=0.99)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")



def scalar_group_vars(idata, group="posterior"):
    if not hasattr(idata, group):
        return []
    ds = getattr(idata, group)
    out = []
    for v in ds.data_vars:
        if v.startswith("prob_"):
            continue
        arr = ds[v]
        if tuple(arr.dims) == ("chain", "draw"):
            out.append(v)
    return out


def scalar_posterior_vars(idata):
    return scalar_group_vars(idata, group="posterior")


def scalar_prior_vars(idata):
    return scalar_group_vars(idata, group="prior")


def nonconstant_vars(idata, vars_in, group="posterior", tol=1e-10):
    if not hasattr(idata, group):
        return []
    ds = getattr(idata, group)

    kept = []
    for v in vars_in:
        if v not in ds:
            continue
        vals = np.asarray(ds[v].values).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if np.nanstd(vals) > tol:
            kept.append(v)
    return kept


def ensure_prior_in_idata(res_or_idata, draws=4000, random_seed=123):
    if hasattr(res_or_idata, "idata"):
        idata = res_or_idata.idata
        model = res_or_idata.model
    else:
        idata = res_or_idata
        model = None

    if hasattr(idata, "prior"):
        return idata

    if model is None:
        raise ValueError("No model attached, so prior samples cannot be generated.")

    with model:
        prior_idata = pm.sample_prior_predictive(
            samples=draws,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return az.concat(idata, prior_idata, dim=None, copy=True)


def _flatten_group_var(idata, group, var_name):
    if not hasattr(idata, group):
        return np.array([])
    ds = getattr(idata, group)
    if var_name not in ds:
        return np.array([])
    vals = np.asarray(ds[var_name].values).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return vals


def _choose_xlim(var_name, post, prior=None, xlim=None, default_pad=0.10):
    if xlim is not None and var_name in xlim:
        lo, hi = xlim[var_name]
        return float(lo), float(hi)

    vals = np.asarray(post, dtype=float)
    vals = vals[np.isfinite(vals)]

    if prior is not None:
        pv = np.asarray(prior, dtype=float)
        pv = pv[np.isfinite(pv)]
        if pv.size:
            vals = np.concatenate([vals, pv]) if vals.size else pv

    if vals.size == 0:
        return (-1.0, 1.0)

    lo = float(np.nanquantile(vals, 0.01))
    hi = float(np.nanquantile(vals, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    pad = default_pad * (hi - lo)
    return lo - pad, hi + pad


def plot_prior_posterior_grouped(
    idata,
    out_path,
    title,
    rows,
    xlim=None,
    bins=60,
    dpi=300,
):
    clean_rows = []
    available_post = set(scalar_posterior_vars(idata))
    for row in rows:
        row_clean = [v for v in row if v in available_post]
        row_clean = nonconstant_vars(idata, row_clean, group="posterior")
        clean_rows.append(row_clean)

    nrows = len(clean_rows)
    ncols = max((len(r) for r in clean_rows), default=0)

    if ncols == 0:
        return None

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    prior_vars = set(scalar_prior_vars(idata))

    for i, row in enumerate(clean_rows):
        for j in range(ncols):
            ax = axes[i, j]

            if j >= len(row):
                ax.axis("off")
                continue

            v = row[j]
            post = _flatten_group_var(idata, "posterior", v)
            prior = _flatten_group_var(idata, "prior", v) if v in prior_vars else np.array([])

            if prior.size:
                ax.hist(
                    prior,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    label="prior",
                )

            if post.size:
                ax.hist(
                    post,
                    bins=bins,
                    density=True,
                    alpha=0.40,
                    label="posterior",
                )
                q025, q50, q975 = np.quantile(post, [0.025, 0.5, 0.975])
                ax.axvline(q50, linestyle="--", linewidth=1.5, label="median")

            ax.set_title(v)
            ax.set_xlim(*_choose_xlim(v, post, prior=prior, xlim=xlim))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path, clean_rows