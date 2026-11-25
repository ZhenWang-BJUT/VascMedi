"""Vascular module response panel."""
import numpy as np
import pandas as pd

def plot_panel(ax, data: pd.DataFrame, title: str, x_label: str) -> None:
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    values = data["signed_iie"].to_numpy(dtype=float)
    ci_low = data["ci_low"].to_numpy(dtype=float)
    ci_high = data["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([values - ci_low, ci_high - values])

    positive = values >= 0
    crosses = data["ci_crosses_zero"].to_numpy(dtype=bool)
    colors = np.where(positive, "#2f6fbb", "#b86b6b")
    alphas = np.where(crosses, 0.42, 0.88)

    for idx in range(len(data)):
        ax.barh(
            y[idx],
            values[idx],
            color=colors[idx],
            alpha=float(alphas[idx]),
            edgecolor="#303030",
            linewidth=0.7,
            height=0.62,
            hatch="//" if crosses[idx] else None,
            zorder=2,
        )

    ax.errorbar(
        values,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#242424",
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        zorder=3,
    )
    ax.axvline(0, color="#5c5c5c", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["module"].tolist(), fontsize=8.8)
    ax.set_title(title, fontsize=10.0, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel(x_label, fontsize=8.5)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, alpha=0.75, zorder=0)
    ax.tick_params(axis="x", labelsize=8.2)
    ax.tick_params(axis="y", length=0)

    xmin = min(float(ci_low.min()), 0.0)
    xmax = max(float(ci_high.max()), 0.0)
    span = xmax - xmin
    ax.set_xlim(xmin - 0.11 * span, xmax + 0.28 * span)
    label_x = xmax + 0.04 * span
    for idx, row in data.iterrows():
        ax.text(
            label_x,
            y[idx],
            "%.2f%%" % row["abs_share_pct"],
            va="center",
            ha="left",
            fontsize=8.2,
            color="#1f1f1f",
        )
    ax.text(
        label_x,
        y.max() + 0.72,
        "|IIE| share",
        va="bottom",
        ha="left",
        fontsize=8.0,
        color="#444444",
    )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#707070")
    ax.spines["bottom"].set_linewidth(0.8)
