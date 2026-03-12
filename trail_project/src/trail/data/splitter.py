"""
Data splitter for TRAIL experiments.

Splits:
  - 2011: entire dataset is the historical prior (train)
  - 2022: test_ratio held out, remaining is pool
           few-shot init drawn from pool at specified ratio
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from trail.utils.logging import get_logger
from trail.utils.seed import set_seed

logger = get_logger(__name__)


def split_2022(
    df_2022: pd.DataFrame,
    test_ratio: float = 0.2,
    fewshot_ratio: float = 0.01,
    seed: int = 42,
    stratify_col: str = "main_mode",
) -> dict[str, pd.DataFrame]:
    """
    Split the 2022 dataset into:
      - test: held-out evaluation set (test_ratio of total)
      - pool: remaining records available for simulated survey
      - few_shot: initial calibration set drawn from pool (fewshot_ratio of pool)

    Stratified sampling by main_mode to preserve class distribution.

    Returns:
        dict with keys: "test", "pool", "few_shot"
    """
    set_seed(seed)

    n_total = len(df_2022)

    # ---- Test split (stratified) ----
    test_indices = _stratified_sample_indices(df_2022, test_ratio, stratify_col, seed)
    df_test = df_2022.iloc[test_indices].copy()
    df_pool_all = df_2022.drop(df_2022.index[test_indices]).copy().reset_index(drop=True)

    # ---- Few-shot split from pool ----
    n_fewshot = max(1, int(len(df_pool_all) * fewshot_ratio))
    fewshot_indices = _stratified_sample_indices(df_pool_all, fewshot_ratio, stratify_col, seed + 1)
    df_fewshot = df_pool_all.iloc[fewshot_indices].copy()
    df_pool = df_pool_all.drop(df_pool_all.index[fewshot_indices]).copy().reset_index(drop=True)

    logger.info(
        f"2022 split (seed={seed}, fewshot={fewshot_ratio:.1%}): "
        f"test={len(df_test):,}, few_shot={len(df_fewshot):,}, pool={len(df_pool):,}"
    )

    return {
        "test": df_test.reset_index(drop=True),
        "pool": df_pool,
        "few_shot": df_fewshot.reset_index(drop=True),
    }


def _stratified_sample_indices(
    df: pd.DataFrame, ratio: float, stratify_col: str, seed: int
) -> list[int]:
    """Return row indices for a stratified sample at the given ratio."""
    rng = np.random.RandomState(seed)
    indices = []
    for label, group in df.groupby(stratify_col):
        n = max(1, int(len(group) * ratio))
        sampled = rng.choice(group.index.tolist(), size=min(n, len(group)), replace=False)
        # Convert label-based index to positional index
        pos_map = {label_idx: pos for pos, label_idx in enumerate(df.index)}
        indices.extend([pos_map[i] for i in sampled if i in pos_map])
    return indices


def make_fewshot_variants(
    df_2022: pd.DataFrame,
    ratios: list[float],
    test_ratio: float = 0.2,
    seeds: list[int] = None,
    stratify_col: str = "main_mode",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Generate splits for multiple few-shot ratios and seeds.

    Returns:
        nested dict: {f"{ratio:.2f}_{seed}": {"test", "pool", "few_shot"}}
    """
    if seeds is None:
        seeds = [42, 123, 456]

    results = {}
    for ratio in ratios:
        for seed in seeds:
            key = f"ratio{ratio:.2f}_seed{seed}"
            results[key] = split_2022(df_2022, test_ratio, ratio, seed, stratify_col)

    return results
