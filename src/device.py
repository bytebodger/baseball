"""Central GPU/device resolution for every training and inference entry point.

This project has only ever been developed and run on one machine, and that
machine has an NVIDIA GPU -- every existing checkpoint, every "tens of
minutes per epoch" timing note in this codebase's docstrings, and every
walk-forward/backtest run to date assumed CUDA. A silent
`"cuda" if torch.cuda.is_available() else "cpu"` default is dangerous here:
if CUDA ever isn't available (the wrong Python interpreter on PATH, a
CPU-only torch wheel shadowing the GPU one, a driver problem), a script would
keep running anyway -- just 10-50x slower, and saving checkpoints on a
different device than the ones already on disk. That's exactly what
happened when this project's global/Microsoft Store Python (CPU-only torch)
shadowed its own .venv (CUDA-enabled torch) on PATH: hours were spent
retraining on CPU before anyone noticed CUDA wasn't actually being used.

`resolve_device` makes that failure loud instead of silent: the default is
always "cuda", and if it's not available, this raises immediately with a
message pointing at the likely cause, instead of quietly falling back.
Passing --device cpu explicitly (small tests, quick local debugging) is
still honored without complaint -- only the *unrequested* absence of CUDA
is an error.
"""

import sys

import torch

DEFAULT_DEVICE = "cuda"


def resolve_device(requested: str) -> torch.device:
    """Raises if `requested` is "cuda" but CUDA isn't actually available.
    Any other value (e.g. "cpu", explicitly requested) is passed through
    unchanged -- this only guards against silently downgrading the default."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but this project trains and backtests on GPU by "
            "default (pass --device cpu to explicitly opt into a CPU run). This almost "
            f"always means the wrong Python interpreter is on PATH -- currently "
            f"{sys.executable}. Activate this project's .venv first "
            "(.venv\\Scripts\\activate on Windows, or run it directly as "
            ".venv\\Scripts\\python.exe) and re-check with: "
            'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
        )
    return torch.device(requested)
