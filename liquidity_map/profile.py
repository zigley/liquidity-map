"""Volume profile and value-area calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    volumes: np.ndarray
    poc_price: float
    vah_price: float
    val_price: float
    hvn_mask: np.ndarray
    lvn_mask: np.ndarray


def _price_bins(low: float, high: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if high <= low:
        high = low + max(abs(low) * 0.001, 0.01)
    edges = np.linspace(low, high, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def build_volume_profile(df: pd.DataFrame, n_bins: int = 100, value_area_pct: float = 0.70) -> VolumeProfile:
    """Distribute bar volume across price bins between low and high."""
    if df.empty:
        raise ValueError("Cannot build volume profile from empty dataframe")

    price_low = float(df["low"].min())
    price_high = float(df["high"].max())
    edges, centers = _price_bins(price_low, price_high, n_bins)
    volumes = np.zeros(n_bins, dtype=float)

    for row in df.itertuples(index=False):
        bar_low = float(row.low)
        bar_high = float(row.high)
        bar_vol = float(row.volume)
        if bar_vol <= 0 or bar_high <= bar_low:
            continue

        lo_idx = int(np.searchsorted(edges, bar_low, side="right") - 1)
        hi_idx = int(np.searchsorted(edges, bar_high, side="left"))
        lo_idx = max(0, min(lo_idx, n_bins - 1))
        hi_idx = max(lo_idx, min(hi_idx, n_bins - 1))

        touched = hi_idx - lo_idx + 1
        per_bin = bar_vol / touched
        volumes[lo_idx : hi_idx + 1] += per_bin

    poc_idx = int(np.argmax(volumes))
    poc_price = float(centers[poc_idx])

    total_vol = volumes.sum()
    if total_vol <= 0:
        return VolumeProfile(
            bin_edges=edges,
            bin_centers=centers,
            volumes=volumes,
            poc_price=poc_price,
            vah_price=price_high,
            val_price=price_low,
            hvn_mask=np.zeros(n_bins, dtype=bool),
            lvn_mask=np.zeros(n_bins, dtype=bool),
        )

    target = total_vol * value_area_pct
    included = {poc_idx}
    accumulated = volumes[poc_idx]
    left = poc_idx - 1
    right = poc_idx + 1

    while accumulated < target and (left >= 0 or right < n_bins):
        left_vol = volumes[left] if left >= 0 else -1
        right_vol = volumes[right] if right < n_bins else -1
        if left_vol >= right_vol:
            if left >= 0:
                included.add(left)
                accumulated += volumes[left]
                left -= 1
            elif right < n_bins:
                included.add(right)
                accumulated += volumes[right]
                right += 1
            else:
                break
        else:
            if right < n_bins:
                included.add(right)
                accumulated += volumes[right]
                right += 1
            elif left >= 0:
                included.add(left)
                accumulated += volumes[left]
                left -= 1
            else:
                break

    idxs = sorted(included)
    vah_price = float(centers[idxs[-1]])
    val_price = float(centers[idxs[0]])

    positive = volumes[volumes > 0]
    median_vol = float(np.median(positive)) if len(positive) else 0.0
    hvn_mask = volumes >= median_vol * 1.5 if median_vol > 0 else np.zeros(n_bins, dtype=bool)
    lvn_mask = (volumes > 0) & (volumes <= median_vol * 0.5) if median_vol > 0 else np.zeros(n_bins, dtype=bool)

    return VolumeProfile(
        bin_edges=edges,
        bin_centers=centers,
        volumes=volumes,
        poc_price=poc_price,
        vah_price=vah_price,
        val_price=val_price,
        hvn_mask=hvn_mask,
        lvn_mask=lvn_mask,
    )