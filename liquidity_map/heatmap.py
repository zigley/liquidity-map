"""Time x price liquidity heatmap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from liquidity_map.profile import VolumeProfile, build_volume_profile


def build_liquidity_heatmap(
    df: pd.DataFrame,
    n_bins: int = 100,
    density: bool = True,
) -> tuple[np.ndarray, list, np.ndarray, VolumeProfile]:
    """
    Build a 2D matrix of liquidity intensity.

    Returns (matrix, x_labels, y_centers, profile) where matrix shape is
    (n_bins, n_bars) with rows = price bins and cols = time.
    """
    profile = build_volume_profile(df, n_bins=n_bins)
    edges = profile.bin_edges
    centers = profile.bin_centers
    n_price_bins = len(centers)
    n_bars = len(df)

    matrix = np.zeros((n_price_bins, n_bars), dtype=float)
    x_labels: list = []

    for col_idx, row in enumerate(df.itertuples(index=False)):
        x_labels.append(row.datetime)
        bar_low = float(row.low)
        bar_high = float(row.high)
        bar_vol = float(row.volume)
        if bar_vol <= 0 or bar_high <= bar_low:
            continue

        if density:
            bar_vol = bar_vol / max(bar_high - bar_low, 1e-9)

        lo_idx = int(np.searchsorted(edges, bar_low, side="right") - 1)
        hi_idx = int(np.searchsorted(edges, bar_high, side="left"))
        lo_idx = max(0, min(lo_idx, n_price_bins - 1))
        hi_idx = max(lo_idx, min(hi_idx, n_price_bins - 1))

        touched = hi_idx - lo_idx + 1
        per_bin = bar_vol / touched
        matrix[lo_idx : hi_idx + 1, col_idx] += per_bin

    return matrix, x_labels, centers, profile