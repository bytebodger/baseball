import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.inference.ensemble_artifacts import EnsembleArtifacts, load_ensemble_artifacts, save_ensemble_artifacts


def test_ensemble_artifacts_module_is_not_a_cli_entry_point():
    """Regression guard: EnsembleArtifacts must live in a module that is
    never run as `python -m ...` (never `__main__`). A class defined inside
    a script that IS run that way gets pickled with module "__main__"
    instead of its real dotted path, so unpickling it later from a
    *different* `-m` entry point (e.g. predict_upcoming.py, itself __main__
    by then) fails with "Can't get attribute 'EnsembleArtifacts' on module
    ...". This bit us once already when the class lived in backtest.py --
    don't move it back there or anywhere else with a `if __name__ ==
    "__main__"` block."""
    assert EnsembleArtifacts.__module__ == "src.inference.ensemble_artifacts"


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((50, 2))
    y = (rng.random(50) > 0.5).astype(int)

    lr_baseline = LogisticRegression().fit(X, y)
    stacking_model = LogisticRegression().fit(X, y)
    artifacts = EnsembleArtifacts(lr_baseline=lr_baseline, stacking_model=stacking_model)

    path = tmp_path / "ensemble_models.pkl"
    save_ensemble_artifacts(artifacts, path)
    loaded = load_ensemble_artifacts(path)

    assert np.array_equal(artifacts.lr_baseline.predict_proba(X), loaded.lr_baseline.predict_proba(X))
    assert np.array_equal(artifacts.stacking_model.predict_proba(X), loaded.stacking_model.predict_proba(X))


def test_load_ensemble_artifacts_survives_a_real_subprocess_round_trip(tmp_path):
    """The actual failure mode: pickle from a script run as `python -m
    some.module` (so EnsembleArtifacts would be tagged "__main__" if it
    lived in that module), then unpickle from a *different* `python -m`
    invocation. Exercises the real bug end-to-end rather than just checking
    __module__ in-process, where every module already has its normal
    dotted name (pytest never runs a module as __main__)."""
    import subprocess
    import sys

    write_script = tmp_path / "write_artifacts.py"
    write_script.write_text(
        "from pathlib import Path\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from src.inference.ensemble_artifacts import EnsembleArtifacts, save_ensemble_artifacts\n"
        "import numpy as np\n"
        "X = np.random.default_rng(0).random((20, 2))\n"
        "y = (np.random.default_rng(1).random(20) > 0.5).astype(int)\n"
        "model = LogisticRegression().fit(X, y)\n"
        f"save_ensemble_artifacts(EnsembleArtifacts(lr_baseline=model, stacking_model=model), Path(r'{tmp_path / 'artifacts.pkl'}'))\n"
    )
    read_script = tmp_path / "read_artifacts.py"
    read_script.write_text(
        "from pathlib import Path\n"
        "from src.inference.ensemble_artifacts import load_ensemble_artifacts\n"
        f"artifacts = load_ensemble_artifacts(Path(r'{tmp_path / 'artifacts.pkl'}'))\n"
        "print('loaded ok')\n"
    )

    subprocess.run([sys.executable, str(write_script)], check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, str(read_script)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "loaded ok" in result.stdout
