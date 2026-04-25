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

    x_vals = plot_df[feature].to_numpy()
    y_vals = plot_df["difficulty"].to_numpy()

    # Highlight isolated points using k-nearest neighbor distance in normalized space.
    """
    if len(x_vals) >= 6:
        x_norm = (x_vals - x_vals.mean()) / (x_vals.std() or 1.0)
        y_norm = (y_vals - y_vals.mean()) / (y_vals.std() or 1.0)
        points = np.column_stack([x_norm, y_norm])
        dists = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        k = 5
        knn = np.sort(dists, axis=1)[:, :k]
        isolation_score = knn.mean(axis=1)
        isolated_idx = np.argsort(isolation_score)[-10:]
    else:
        isolated_idx = np.array([], dtype=int)
    """
        
    sns.regplot(
        data=plot_df,
        x=feature,
        y="difficulty",
        scatter_kws={"s": 8, "alpha": 0.7},
        line_kws={"color": "red"},
        ax=ax,
    )
    

    """if isolated_idx.size:
        for idx in isolated_idx:
            ax.annotate(
                plot_df.iloc[idx]["song_id"],
                (x_vals[idx], y_vals[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )"""

    ax.set_title(f"{feature} vs difficulty")

for ax in axes[len(features):]:
    ax.axis("off")

plt.tight_layout()
plt.show()