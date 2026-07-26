"""Serving path: chunks -> one bot-risk score per chunk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from detector.features import behavior_features, profile_features

_ART = Path(__file__).resolve().parent / "artifacts"

# Defined here (not a separate module) because model.joblib pickles instances of
# these classes, so they must import from a shipped file to unpickle at serve
# time. train.py imports them from here too.

MEMBER_WEIGHTS = {"stack": 0.30, "mono": 0.20, "mlp": 0.30, "subspace": 0.20}


class SubModel:
    """Bagged trees over random feature subspaces and bootstrap rows."""

    def __init__(self, n=10, ff=0.7, seed=0):
        self.n, self.ff, self.seed = n, ff, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.seed)
        nf = X.shape[1]
        k = max(1, int(self.ff * nf))
        self.mem = []
        for b in range(self.n):
            fi = np.sort(rng.choice(nf, k, replace=False))
            rows = rng.choice(len(X), len(X), replace=True)
            m = (ExtraTreesClassifier(300, max_depth=14, min_samples_leaf=2, n_jobs=4,
                                      random_state=b, class_weight="balanced_subsample")
                 if b % 2 == 0 else
                 HistGradientBoostingClassifier(max_depth=4, learning_rate=0.04, max_iter=350,
                                                l2_regularization=2.0, random_state=b))
            m.fit(X[np.ix_(rows, fi)], y[rows])
            self.mem.append((fi, m))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        P = np.column_stack([m.predict_proba(X[:, fi])[:, 1] for fi, m in self.mem])
        a = P.mean(1)
        return np.column_stack([1.0 - a, a])


class Ensemble:
    """Weighted blend of the member models."""

    def __init__(self, stack, mono, mlp, subspace, cols_profile, cols_behavior, weights=None):
        self.stack = stack
        self.mono = mono
        self.mlp = mlp
        self.subspace = subspace
        self.cols_profile = cols_profile
        self.cols_behavior = cols_behavior
        self.weights = dict(weights) if weights else dict(MEMBER_WEIGHTS)

    @staticmethod
    def _rank(s):
        s = np.asarray(s, dtype=float)
        n = s.size
        if n <= 1:
            return s
        return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (n - 1)

    def score(self, x_profile, x_behavior):
        """Return a blended score in [0, 1] for each row (higher = more bot-like)."""
        x_profile = np.asarray(x_profile, float)
        x_behavior = np.asarray(x_behavior, float)
        x_union = np.hstack([x_behavior, x_profile])
        w = self.weights
        r = (w["stack"] * self._rank(self.stack.predict_proba(x_profile)[:, 1])
             + w["mono"] * self._rank(self.mono.predict_proba(x_profile)[:, 1])
             + w["mlp"] * self._rank(self.mlp.predict_proba(x_union)[:, 1])
             + w["subspace"] * self._rank(self.subspace.predict_proba(x_behavior)[:, 1]))
        return r / sum(w.values())


POSITIVE_FRACTION = float(os.environ.get("POKER44_POSITIVE_FRACTION", "0.05"))

# Below this, a batch is too small for a fraction to be meaningful; fall back to
# the threshold fitted at training time.
_MIN_BATCH = 8


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone map sending t -> 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def place_threshold(p: np.ndarray, fraction: float = POSITIVE_FRACTION,
                    fallback: float = 0.5) -> np.ndarray:
    """Rescale scores so the top ``fraction`` of the batch sits >= 0.5."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    if n < _MIN_BATCH:
        return _remap_to_threshold(p, fallback)
    k = max(1, int(round(fraction * n)))
    k = min(k, n)
    cut = float(np.sort(p)[::-1][k - 1])
    return _remap_to_threshold(p, cut)


class Detector:
    """Loads the trained model and scores validator batches."""

    def __init__(self, art_dir: Path | str = _ART):
        art_dir = Path(art_dir)
        self.ens: Ensemble = joblib.load(art_dir / "model.joblib")
        with open(art_dir / "meta.json") as fh:
            self.meta = json.load(fh)
        self.fallback_threshold: float = float(self.meta.get("deploy_threshold", 0.5))
        self.cols_profile = self.ens.cols_profile
        self.cols_behavior = self.ens.cols_behavior

    def _matrices(self, chunks):
        prof = np.array([[float(d.get(c, 0.0)) for c in self.cols_profile]
                         for d in (profile_features(c) for c in chunks)], dtype=float)
        beh = np.array([[float(d.get(c, 0.0)) for c in self.cols_behavior]
                        for d in (behavior_features(c) for c in chunks)], dtype=float)
        return prof, beh

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        prof, beh = self._matrices(chunks)
        p = self.ens.score(prof, beh)
        scores = place_threshold(p, POSITIVE_FRACTION, self.fallback_threshold)
        return [0.1 if not chunk else round(float(s), 6)
                for chunk, s in zip(chunks, scores)]


_SINGLETON: Detector | None = None


def get_model() -> Detector:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Detector()
    return _SINGLETON
