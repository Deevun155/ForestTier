# a quick script to visualise the data

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_path = Path(__file__).resolve().parent.parent / "hmx_dataset.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found at {csv_path}")

df = pd.read_csv(csv_path)

# Feature vs difficulty plots
features = [
    "total_active_time",
    "avg_nps",
    "peak_nps",
    "avg_strums_per_sec",
    "peak_strums_per_sec",
    "avg_fret_changes_per_sec",
    "peak_fret_changes_per_sec",
]

cols = 3
rows = (len(features) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    plot_df = df[["song_id", feature, "difficulty"]].dropna()
    if plot_df.empty:
        ax.set_title(f"{feature} vs difficulty (no data)")
        ax.axis("off")
        continue

    sns.regplot(
        data=plot_df,
        x=feature,
        y="difficulty",
        scatter_kws={"s": 8},
        line_kws={"color": "red"},
        ax=ax,
    )
    ax.set_title(f"{feature} vs difficulty")

    x_vals = plot_df[feature].to_numpy()
    y_vals = plot_df["difficulty"].to_numpy()
    if len(x_vals) >= 2:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        preds = slope * x_vals + intercept
        residuals = np.abs(y_vals - preds)
        top_idx = np.argsort(residuals)[-10:]
        for idx in top_idx:
            ax.annotate(
                plot_df.iloc[idx]["song_id"],
                (x_vals[idx], y_vals[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

for ax in axes[len(features):]:
    ax.axis("off")

plt.tight_layout()
plt.show()