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
- On this project's g6.xlarge AWS instances (Deep Learning AMI setup used for encoder/event-model training),
  `~/baseball` lives on the persistent EBS root volume, not the ephemeral local NVMe instance store -- confirmed
  directly (2026-07) by recovering a checkpoint from a stopped instance's root-volume snapshot via a separate
  helper instance. Stop/start cycles are safe for this directory specifically and don't risk losing anything
  under it. Doesn't need re-verifying on future instances built from the same AMI/setup.
- `src/data/event_embedding_cache.py`'s `EmbeddingCache` memoizes every (player_id, game_date) embedding it
  looks up in memory and never evicts by default (`max_entries=None`) -- confirmed (2026-07) to plateau around
  ~17.7KB/entry, ~15GB for boundary 2's ~687K-pair scale, growing with dataset size boundary-over-boundary as
  this project's walk-forward validation moves through later season boundaries. Unbounded is fine as long as
  it's been checked against the machine it's running on (confirmed comfortable on this project's 32GB local
  dev machine through boundary 2); pass `--embedding-cache-max-entries` (wired through
  `train_event_model.py`) once a boundary's estimated total distinct (player, date) pairs would push unbounded
  memory close to the machine's ceiling. Whatever cap is chosen must comfortably *exceed* the run's total
  distinct pairs, not undercut it as a "smaller footprint" -- an isolated test (see
  `EmbeddingCache`'s docstring) demonstrated that an undersized budget causes severe, measurable thrashing
  (repeated evict-then-immediately-reload cycles: hit rate collapsed to ~2% and wall-clock ran ~2.8x slower at
  25% of the real working set), not a graceful degradation, so treat the cap as a safety ceiling sized above
  the expected need, never as a routine reduction mechanism.