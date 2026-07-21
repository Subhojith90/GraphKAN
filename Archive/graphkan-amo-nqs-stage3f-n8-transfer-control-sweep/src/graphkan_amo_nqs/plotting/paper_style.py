from __future__ import annotations

import matplotlib.pyplot as plt


def apply_paper_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.figsize": (5.2, 3.2),
        "savefig.bbox": "tight",
    })
