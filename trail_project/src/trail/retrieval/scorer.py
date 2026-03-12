"""
Retrieval scoring functions.

Score(x, j) = λ1 * Sim(x, j) + λ2 * Shift(j) + λ3 * Priority(j)

For MVP-1:
  - Sim: cosine similarity on normalized tabular features
  - Shift: temporal shift weight (2022 > 2011 for recency)
  - Priority: calibration error priority (disabled in MVP-1)
"""

from typing import Optional

import numpy as np
import pandas as pd


def cosine_similarity_matrix(query_X: np.ndarray, corpus_X: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vectors and corpus vectors.

    Args:
        query_X: (n_queries, n_features) normalized feature matrix
        corpus_X: (n_corpus, n_features) normalized feature matrix

    Returns:
        (n_queries, n_corpus) similarity matrix in [-1, 1]
    """
    # L2-normalize
    q_norm = np.linalg.norm(query_X, axis=1, keepdims=True)
    c_norm = np.linalg.norm(corpus_X, axis=1, keepdims=True)

    q_norm = np.where(q_norm == 0, 1.0, q_norm)
    c_norm = np.where(c_norm == 0, 1.0, c_norm)

    Q = query_X / q_norm
    C = corpus_X / c_norm

    return Q @ C.T  # (n_queries, n_corpus)


def similarity_score(
    query_X: np.ndarray, corpus_X: np.ndarray
) -> np.ndarray:
    """Cosine similarity (batch)."""
    return cosine_similarity_matrix(query_X, corpus_X)


def shift_score(
    corpus_df: pd.DataFrame,
    survey_year_col: str = "survey_year",
    target_year: int = 2022,
) -> np.ndarray:
    """
    Temporal shift weight: 2022 records get weight 1.0, 2011 records get 0.5.
    Records closer in time to the target year are preferred.
    """
    years = corpus_df[survey_year_col].fillna(2011).values
    weights = np.where(years >= target_year, 1.0, 0.5)
    return weights  # (n_corpus,)


def priority_score(
    corpus_df: pd.DataFrame,
    calibration_errors: Optional[dict] = None,
) -> np.ndarray:
    """
    Calibration error priority. Returns uniform 1.0 in MVP-1.
    Will be populated in Phase D (active survey loop).
    """
    return np.ones(len(corpus_df))


def compute_retrieval_scores(
    query_X: np.ndarray,
    corpus_X: np.ndarray,
    corpus_df: pd.DataFrame,
    lambda_sim: float = 0.5,
    lambda_shift: float = 0.3,
    lambda_priority: float = 0.2,
    calibration_errors: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute combined retrieval scores for all (query, corpus) pairs.

    Returns:
        (n_queries, n_corpus) score matrix
    """
    sim = similarity_score(query_X, corpus_X)  # (n_q, n_c)
    shift = shift_score(corpus_df)              # (n_c,)
    priority = priority_score(corpus_df, calibration_errors)  # (n_c,)

    scores = (
        lambda_sim * sim +
        lambda_shift * shift[np.newaxis, :] +
        lambda_priority * priority[np.newaxis, :]
    )
    return scores
