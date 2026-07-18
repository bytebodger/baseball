# Project instructions

- Write tests as you go — add/update tests alongside the code change, not as a separate follow-up pass.
- Keep configs in YAML files.
- Always run Python via this project's `.venv` (`.venv\Scripts\python.exe` on Windows, or `.venv\Scripts\activate`
  first), never the global/Microsoft Store Python. The `.venv` has the CUDA-enabled PyTorch build for this
  machine's GPU; the global interpreter only has a CPU-only build. This project trains/backtests on GPU by
  default and `src/device.py`'s `resolve_device` will raise loudly if CUDA isn't available -- if you hit that
  error, you're on the wrong interpreter, not a machine without a GPU.
- Any job expected to run longer than roughly 30 minutes (batched simulation studies, large cache-build/backfill
  jobs, validation studies) must checkpoint incrementally to disk as it works, not hold results in memory until a
  final write, and must be safely resumable (skip already-completed work on rerun). Drive it with
  `src/resumable_job.py`'s `run_until_complete`, which repeatedly invokes the job until its progress file (see
  `write_progress`/`read_progress`) reports zero remaining work, logging every attempt and raising instead of
  looping silently if 10 consecutive attempts show no forward progress *or* no plausible chance of finishing in
  reasonable time (pass `min_rate_items_per_second` and/or `expected_completion_seconds` -- a job with nonzero
  but too-slow progress is flagged the same way a fully stalled one is, not allowed to run for days). This exists
  because background shell jobs in this environment have an observed (undocumented) duration ceiling around
  45-55 minutes -- see that module's docstring for the incident this came from.
- After any retrain of the event model (`src/training/train_event_model.py`), run the aggregate
  marginal-probability check (compare the model's average predicted probability per `OUTCOME_VOCAB` category,
  over a large sample of real held-out situations, against the real observed marginal frequency for each
  category) as the *first* validation step, before any simulation-based check. Only proceed to full-game
  simulation and paired-pitcher probes once marginal calibration looks reasonable. This exists because a
  training-loss change (inverse-frequency class weighting) once passed this project's paired-pitcher and
  low-scoring-game probes' *relative* comparisons while being badly broken in *absolute* terms (predicted
  extra-base-hit rate ~3x real, ~59 simulated runs/game vs. a real ~9) -- the marginal check catches that class
  of failure immediately and cheaply (seconds, not a 10-minute-plus full-game simulation study) before sinking
  time into simulation-based validation that would have been built on a miscalibrated model anyway.