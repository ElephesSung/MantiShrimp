#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "legend.fontsize": 10,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})

non_interaction = [
    *[np.array([]) for _ in range(38)],
]
no_kill = [
    *[np.array([0]) for _ in range(23)],
    *[np.array([0, 0]) for _ in range(13)],
    *[np.array([0, 0, 0]) for _ in range(10)],
    *[np.array([0, 0, 0, 0]) for _ in range(2)],
    *[np.array([0, 0, 0, 0, 0]) for _ in range(1)],
    *[np.array([0, 0, 0, 0, 0, 0]) for _ in range(1)]
]
kill = [
    *[np.array([1]) for _ in range(14)],
    *[np.array([1, 1]) for _ in range(15)],
    *[np.array([1, 1, 1]) for _ in range(13)],
    *[np.array([1, 1, 1, 1]) for _ in range(5)],
    *[np.array([1, 1, 1, 1, 1]) for _ in range(3)],
    *[np.array([1, 1, 1, 1, 1, 1]) for _ in range(3)],
    *[np.array([1, 1, 1, 1, 1, 1, 1]) for _ in range(1)]
]
exhausted = [
    *[np.array([1, 0]) for _ in range(7)],
    *[np.array([1, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 0]) for _ in range(9)],
    *[np.array([1, 0, 0, 0]) for _ in range(2)],
    *[np.array([1, 1, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 0]) for _ in range(1)],
    *[np.array([1, 1, 0, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 0, 0]) for _ in range(3)],
    *[np.array([1, 1, 1, 1, 0]) for _ in range(1)],
    *[np.array([1, 0, 0, 0, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 1, 1, 0]) for _ in range(2)],
    *[np.array([1, 1, 1, 1, 1, 1, 1, 0]) for _ in range(1)],
]
stochastic = [
    *[np.array([0, 1, 1, 0]) for _ in range(1)],
    *[np.array([1, 0, 1, 0]) for _ in range(1)],
    *[np.array([1, 1, 0, 1, 0]) for _ in range(1)],
    *[np.array([1, 0, 1, 0, 0, 0]) for _ in range(1)],
    *[np.array([0, 0, 0, 1, 1, 1]) for _ in range(1)],
    *[np.array([0, 1, 1, 0, 0, 0, 0]) for _ in range(1)],
]
total_cells = len(no_kill) + len(kill) + len(exhausted) + len(stochastic)
print(
    f"Data from Blood (2013) 121 (8): 1326–1334. Total number of cells represented: {total_cells}","\n", 
    f"non interaction NK cells: number - {len(non_interaction)}; proportion - {np.divide(len(non_interaction), total_cells)}", "\n",
    f"non kill NK cells: number - {len(no_kill)}; proportion - {np.divide(len(no_kill), total_cells)}", "\n",
    f"stochastic NK cells: number - {len(stochastic)}; proportion - {np.divide(len(stochastic), total_cells)}", "\n",
    f"exhausted NK cells: number - {len(exhausted)}; proportion - {np.divide(len(exhausted), total_cells)}", "\n",
    f"killing NK cells: number - {len(kill)}; proportion - {np.divide(len(kill), total_cells)}", "\n",
    )
group_data = [
    (non_interaction, "No interaction", "#0064B2"),
    (no_kill, "No kill", "#547976"),
    (exhausted, "Exhausted", "#8C874E"),
    (stochastic, "Stochastic", "#C49526"),
    (kill, "Serial-killing", "#ffa200"),
]

def decision_map(
    decision_data_group,
    figsize=(6, 6),
    dpi = 300,
    offset_scale = 7,
    alpha = 0.6,
    lw = 2,
    s = 18,
    zorder = 5,
    edgecolor = 'black',
    PDF_path = None,
    PNG_path = None,
    ):
    plt.figure(figsize=figsize, dpi = dpi)
    max_x, max_y = 0, 0
    offset_scale = offset_scale
    
    for group_idx, (group, label, color) in enumerate(decision_data_group):
        n_cells = len(group)
        for i, arr in enumerate(group):
            x, y = [0], [0]
            for val in arr:
                if val  == 1:
                    x.append(x[-1])
                    y.append(y[-1] + 1)
                elif val == 0:
                    x.append(x[-1] + 1)
                    y.append(y[-1])
            offset = (i - n_cells / 2) / (offset_scale * n_cells)
            x = np.array(x) + offset
            y = np.array(y) + offset
            plt.plot(x, y, alpha=alpha, color=color, lw=lw)
            plt.scatter(x[-1], y[-1], color=color, s=s, zorder=zorder, edgecolor=edgecolor)
            max_x = max(max_x, x[-1])
            max_y = max(max_y, y[-1])
        plt.plot([], [], color=color, label=label, lw=10)
    plt.xlabel("Non-lethal contacts")
    plt.ylabel("Lethal contacts")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(1)) 
    ax.yaxis.set_major_locator(MultipleLocator(1)) 
    plt.savefig(PDF_path) if PDF_path else None
    plt.savefig(PNG_path) if PNG_path else None
    # plt.show()

def main():
    decision_map(
        decision_data_group=group_data,
        figsize=(6, 6),
        dpi = 300,
        PNG_path="caterpillar_visualization.png",
        PDF_path="caterpillar_visualization.pdf",
        offset_scale= 4,
    )

if __name__ == "__main__":
    main()