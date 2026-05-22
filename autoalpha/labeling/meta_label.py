"""Meta-labeling: secondary classifier to filter primary signal bets.

Per López de Prado (Advances in Financial ML, Ch. 3):
  1. A primary model gives direction (+1 / -1).
  2. A meta-model learns whether the primary signal will be correct in each case.
  3. Bet size = meta-model confidence in (primary signal being correct).

Critical constraint: the meta-model MUST be fit on each CPCV fold's in-sample
data and applied to that fold's OOS data — never train globally and apply.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetaLabeler:
    """Wraps a scikit-learn classifier for meta-labeling.

    fit()           — train on in-sample data for one CPCV fold
    predict_proba() — return bet-size series (P(primary correct)) for OOS data
    """

    def __init__(self, classifier: Optional[object] = None):
        self._clf = classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._fitted = False
        self._fallback_proba = 1.0  # overridden if fit sees degenerate labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        primary_labels: pd.Series,
        triple_barrier_labels: pd.Series,
    ) -> "MetaLabeler":
        """Train meta-model on in-sample data for one CPCV fold.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix indexed by date.
        primary_labels : pd.Series
            Primary signal direction (+1 / -1), indexed by date.
        triple_barrier_labels : pd.Series
            Actual outcome from triple-barrier labeling (+1 / 0 / -1), indexed by date.

        The meta-label is 1 when sign(primary) == sign(outcome), else 0.
        """
        common = (
            features.index
            .intersection(primary_labels.index)
            .intersection(triple_barrier_labels.index)
        )
        if len(common) < 10:
            logger.warning(
                "Only %d common observations for meta-labeling fit — skipping", len(common)
            )
            return self

        X = features.loc[common]
        primary = primary_labels.loc[common]
        outcome = triple_barrier_labels.loc[common]

        # Meta-label: 1 when primary signal was correct
        meta_label = (np.sign(primary.values) == np.sign(outcome.values)).astype(int)

        if meta_label.sum() == 0 or (1 - meta_label).sum() == 0:
            self._fallback_proba = float(meta_label.mean())
            logger.warning("Degenerate meta-labels (all one class) — skipping fit")
            return self

        X_scaled = self._scaler.fit_transform(X.values)
        self._clf.fit(X_scaled, meta_label)  # type: ignore[union-attr]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Return P(primary signal correct) as bet size in [0, 1].

        Falls back to 1.0 (full bet) when not fitted.
        """
        if not self._fitted:
            return pd.Series(self._fallback_proba, index=features.index)

        X_scaled = self._scaler.transform(features.values)
        proba = self._clf.predict_proba(X_scaled)  # type: ignore[union-attr]

        classes = list(self._clf.classes_)  # type: ignore[union-attr]
        if 1 not in classes:
            return pd.Series(0.0, index=features.index)

        pos_idx = classes.index(1)
        return pd.Series(proba[:, pos_idx], index=features.index)
