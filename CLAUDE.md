# Project instructions

- Write tests as you go — add/update tests alongside the code change, not as a separate follow-up pass.
- Keep configs in YAML files.
- Always run Python via this project's `.venv` (`.venv\Scripts\python.exe` on Windows, or `.venv\Scripts\activate`
  first), never the global/Microsoft Store Python. The `.venv` has the CUDA-enabled PyTorch build for this
  machine's GPU; the global interpreter only has a CPU-only build. This project trains/backtests on GPU by
  default and `src/device.py`'s `resolve_device` will raise loudly if CUDA isn't available -- if you hit that
  error, you're on the wrong interpreter, not a machine without a GPU.