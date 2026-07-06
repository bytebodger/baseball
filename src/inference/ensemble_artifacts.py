"""EnsembleArtifacts lives in its own module, never a CLI entry point,
specifically so it pickles/unpickles correctly across processes. A class
defined inside a script that's run via `python -m some.module` gets tagged
with module "__main__" when pickled (that's what `__name__` resolves to for
whichever script is currently the entry point) -- so an object pickled by
`backtest.py`'s CLI run and later unpickled inside `predict_upcoming.py`'s
CLI run would fail with "Can't get attribute 'EnsembleArtifacts' on module
...predict_upcoming" (predict_upcoming is now __main__, and it never defined
that class). Keeping the class in a plain, never-`__main__` module sidesteps
the problem entirely: its pickled module reference is always this file's
real dotted path, regardless of which script imports and uses it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LogisticRegression


@dataclass
class EnsembleArtifacts:
    """Everything predict_upcoming.py needs to reproduce backtest.py's
    ensemble on new games without refitting anything: the ERA/wRC+ logistic
    regression baseline itself, and the stacking model that blends it with
    GamePredictor."""

    lr_baseline: LogisticRegression
    stacking_model: LogisticRegression


def save_ensemble_artifacts(artifacts: EnsembleArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)


def load_ensemble_artifacts(path: Path) -> EnsembleArtifacts:
    with open(path, "rb") as f:
        return pickle.load(f)
