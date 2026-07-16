"""Standing convention for any job expected to run longer than roughly 30
minutes (batched simulation studies, large cache-build/backfill jobs,
future validation studies): checkpoint incrementally to disk, and resume
automatically rather than relying on a person noticing a silent stop and
manually rerunning the same command.

This convention exists because of a real incident: extending the event
embedding cache across the full 2015-2026 dataset (~806K (pitcher, date)
pairs) got silently killed twice, each time roughly 45-55 minutes into a
background shell invocation, with no error and no documented cause found
anywhere in this environment's own tool descriptions -- only two empirical
data points suggesting a background-task duration ceiling somewhere in that
range. The job survived both kills only because
src/data/event_embedding_cache.py's precompute_and_cache_embeddings already
happened to write each entry to disk immediately (built for crash-tolerance
in general, not for this specific scenario) -- recovering required a human
noticing the process had stopped and manually rerunning the identical
command three separate times. That should have been automatic.

Two halves to this convention:

1. **The job's own responsibility** (not enforced by this module -- a job
   author has to actually do it): write real output incrementally as work
   completes, not held in memory until a single final write, AND call
   write_progress() after each batch/chunk/game so an external wrapper (or
   a human) can see how much work remains without re-deriving it from the
   job's own domain-specific output. A job that satisfies this is safe to
   kill and rerun at any point -- rerunning must skip already-completed
   work (the same "check what's already done, only compute what's missing"
   pattern src/data/event_embedding_cache.py and
   src/data/sequence_dataset.py's precompute_and_cache already use).

2. **run_until_complete()**: a thin, job-agnostic wrapper that repeatedly
   invokes a resumable job (a subprocess command) until its progress file
   reports zero remaining work, logging every attempt (timestamp + remaining
   count) and raising a loud RuntimeError instead of looping silently
   forever if `max_consecutive_stalls` attempts pass with no forward
   progress -- a genuinely stuck job (not just one that keeps hitting the
   environment's duration ceiling) should surface as an error, not an
   infinite quiet retry loop.

Given the ~45-55 minute empirical ceiling is a real but unconfirmed
constraint (nothing in this environment documents it -- see the
resumable-job design discussion this module was born from), the safest
usage pattern is: run *one bounded attempt per external invocation*
(`max_attempts=1`, the default) with `attempt_timeout_seconds` comfortably
under that ceiling (1800s / 30 minutes by default), and let an outer
scheduling loop -- an agent's own self-rescheduling wakeup loop, or
eventually a cron-style schedule -- invoke run_until_complete again rather
than trusting a single long-lived process to internally loop across the
job's full multi-hour duration. `max_attempts=None` (loop internally until
done or stalled) is only safe for jobs known to finish well under that
ceiling.

Stall/rate state (consecutive-stall count, attempt history) is persisted to
a companion `<progress_path>.attempts.jsonl` file, not held in
run_until_complete's own process memory -- an earlier version got this
wrong: with in-memory-only state, the recommended `max_attempts=1` usage
pattern (a fresh process per attempt) meant every single invocation saw
`last_completed=None` and treated itself as the first attempt ever, so
stall detection silently never accumulated across attempts in exactly the
mode this module recommends as safest. Reading history back from disk at
the start of every call is what makes detection actually work regardless
of whether one process loops internally or a scheduler launches a fresh
one per attempt.

Two ways a job can be flagged, both counted against the same
`max_consecutive_stalls` threshold (surfaced the same way, since both mean
"this isn't going to finish in a reasonable amount of time" -- there's no
practical difference to a caller between a job that's stopped and one
that's crawling too slowly to matter):

- **Zero or negative forward progress** between consecutive attempts (the
  original check).
- **Nonzero but too-slow progress**, if the caller supplies
  `min_rate_items_per_second` and/or `expected_completion_seconds` (which
  implies a minimum rate given the job's own `total`): the *recent*
  completion rate (over the last `rate_window_attempts` attempts, not an
  all-time average) is compared against the required minimum. Windowed
  rather than all-time-average deliberately -- the real incident this
  module is modeled on wasn't a hard stall, it was a steadily decelerating
  rate (~600 pairs/s down to ~25 pairs/s as the job worked through
  longer-history players); an all-time average would have stayed
  healthy-looking for a long time after the job had already become
  impractically slow, while a windowed rate reflects what's happening
  *right now*.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_STALLS = 10
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 1800  # 30 minutes -- comfortably under the observed ~45-55 minute ceiling.


@dataclass
class JobProgress:
    total: int
    completed: int
    remaining: int
    updated_at: str  # ISO 8601, UTC
    extra: dict | None = None


def write_progress(path: Path, total: int, completed: int, extra: dict | None = None) -> None:
    """Call this from inside a long job after each batch/chunk/game
    completes. Overwrites `path` with the latest snapshot -- callers don't
    need append/merge semantics, just "what does progress look like right
    now." `total`/`completed` must be in whatever unit the job's own work is
    naturally counted in (games simulated, pairs computed, ...); this module
    has no opinion on what that unit is.

    Written via a temp file + os.replace rather than a direct
    `path.write_text(...)` -- a plain overwrite killed mid-write (the exact
    scenario this whole module exists to survive) could leave `path`
    containing a truncated, unparseable mix of old and new bytes, losing
    *all* progress rather than just the latest update (unlike the
    append-only attempt log, where the same kind of kill can only ever
    corrupt the most recent, still-unflushed line -- see
    _read_attempt_log). os.replace's rename is atomic on both POSIX and
    Windows: a reader of `path` always sees either the complete old content
    or the complete new content, never a partial mix.
    """
    progress = JobProgress(
        total=total,
        completed=completed,
        remaining=max(total - completed, 0),
        updated_at=datetime.now(timezone.utc).isoformat(),
        extra=extra,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(asdict(progress)))
    os.replace(tmp_path, path)


def read_progress(path: Path) -> JobProgress | None:
    """None if `path` doesn't exist yet -- a job that hasn't written its
    first progress snapshot yet, not an error."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return JobProgress(**data)


@dataclass
class AttemptRecord:
    timestamp: str  # ISO 8601, UTC
    completed: int
    total: int
    consecutive_stalls: int  # the running count *after* this attempt was evaluated


def _attempt_log_path(progress_path: Path) -> Path:
    return progress_path.with_name(progress_path.name + ".attempts.jsonl")


def _read_attempt_log(progress_path: Path) -> list[AttemptRecord]:
    """Each _append_attempt_record call opens, writes exactly one line, and
    closes -- so a kill mid-write can only ever leave the *trailing* line
    truncated/corrupt; every earlier line was already fully flushed and
    closed by a prior, distinct call before this one even started. A
    corrupt trailing line is therefore an expected, recoverable failure
    mode: it's dropped (with a loud warning, not silently) and treated as
    if that attempt's record was never written, same as if the kill had
    landed one instruction earlier. A corrupt line anywhere *other* than
    the last one has no such benign explanation (disk corruption, manual
    editing, a bug) and is refused rather than silently skipped -- dropping
    an arbitrary middle record could quietly understate consecutive_stalls
    or reconstruct a wrong last_completed, which is worse than crashing.
    """
    log_path = _attempt_log_path(progress_path)
    if not log_path.exists():
        return []

    lines = log_path.read_text().splitlines()
    nonblank_indices = [i for i, line in enumerate(lines) if line.strip()]
    last_nonblank = nonblank_indices[-1] if nonblank_indices else None

    records = []
    for i in nonblank_indices:
        try:
            records.append(AttemptRecord(**json.loads(lines[i])))
        except (json.JSONDecodeError, TypeError) as exc:
            if i == last_nonblank:
                logger.warning(
                    "Trailing line %d/%d in %s is truncated/corrupt (%s) -- treating it as if that attempt's "
                    "record was never written (likely killed mid-write) and continuing with the %d prior, "
                    "valid record(s).",
                    i + 1, len(lines), log_path, exc, len(records),
                )
                break
            raise RuntimeError(
                f"Corrupt attempt-log line {i + 1}/{len(lines)} in {log_path}, and it is NOT the trailing "
                "line -- every earlier line should already have been safely flushed and closed before any "
                "later write began, so this indicates something worse than an ordinary kill-mid-write (disk "
                "corruption, a manually edited file, a bug in this module). Refusing to silently reconstruct "
                f"stall history from a log with an unexplained gap: {exc}. Inspect or delete {log_path} "
                "directly before resuming."
            ) from exc
    return records


def _append_attempt_record(progress_path: Path, record: AttemptRecord) -> None:
    log_path = _attempt_log_path(progress_path)
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def _update_stall_count(completed: int, last_completed: int | None, consecutive_stalls: int, too_slow: bool = False) -> int:
    """The new consecutive-stall count given this attempt's `completed`
    versus the previous attempt's, plus whether the recent completion rate
    was independently judged too slow (see _is_too_slow). The very first
    attempt (`last_completed is None`) can still count as a stall via
    `too_slow` -- unlike the no-progress check, a rate judgement doesn't
    need a *previous* attempt to compare against, only enough history
    within the window (see _recent_rate) to compute a rate at all, which by
    construction can't exist until at least 2 attempts have happened
    regardless of which one of those is "the first"."""
    no_progress = last_completed is not None and completed <= last_completed
    if no_progress or too_slow:
        return consecutive_stalls + 1
    return 0


def _recent_rate(records: list[AttemptRecord], window: int) -> float | None:
    """Items-per-second over the most recent `window` attempts (or all
    available if fewer than `window`) -- None if there's fewer than 2
    records to measure a rate between, or if no wall-clock time actually
    elapsed (shouldn't happen for real attempts, guarded rather than
    dividing by zero)."""
    if len(records) < 2:
        return None
    recent = records[-window:] if len(records) > window else records
    first, last = recent[0], recent[-1]
    elapsed = (datetime.fromisoformat(last.timestamp) - datetime.fromisoformat(first.timestamp)).total_seconds()
    if elapsed <= 0:
        return None
    return (last.completed - first.completed) / elapsed


def _is_too_slow(records: list[AttemptRecord], min_rate: float | None, window: int) -> bool:
    if min_rate is None:
        return False
    rate = _recent_rate(records, window)
    if rate is None:
        return False
    return rate < min_rate


def run_until_complete(
    command: list[str],
    progress_path: Path,
    max_consecutive_stalls: int = DEFAULT_MAX_CONSECUTIVE_STALLS,
    attempt_timeout_seconds: int | None = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    max_attempts: int | None = 1,
    min_rate_items_per_second: float | None = None,
    expected_completion_seconds: float | None = None,
    rate_window_attempts: int = 3,
) -> JobProgress:
    """Repeatedly invokes `command` (argv list, no shell=True) until
    `progress_path` reports remaining == 0, logging every attempt (ISO
    timestamp + remaining-work count). `command` must be the exact same
    resumable job each time -- this wrapper has no idea what the job does,
    only what its progress file says. Stall/rate state is read from and
    appended to `<progress_path>.attempts.jsonl` at the start/end of every
    call, not held only in this process's memory -- see module docstring
    for why that matters under the recommended max_attempts=1 usage pattern.

    A subprocess that hits `attempt_timeout_seconds` is terminated and
    treated as a normal, expected outcome (not an error) -- that's the
    whole point of bounding each attempt below this environment's observed
    background-task ceiling. A subprocess that exits with a nonzero
    returncode is logged as a warning (distinct from a clean timeout, since
    it likely means a real bug rather than just running out of time) but
    still only feeds into the same stall-detection logic below, rather than
    raising immediately -- one crashed attempt that still made partial
    progress before dying is not distinguishable, from this wrapper's
    perspective, from one that hit its timeout after making the same
    partial progress, and both deserve the same "try again" treatment.

    Stops and raises RuntimeError, rather than looping silently forever, if
    `max_consecutive_stalls` attempts in a row show no increase in
    `completed`, OR (if `min_rate_items_per_second` and/or
    `expected_completion_seconds` are given) show a recent completion rate
    below the required minimum -- a job that's technically still moving but
    far too slowly to plausibly finish gets treated the same as one that's
    genuinely stuck, not allowed to run for days just because "remaining"
    ticks down by one every so often. `expected_completion_seconds` is
    convenience sugar: it's converted to `progress.total / expected_completion_seconds`
    and combined with `min_rate_items_per_second` (whichever requires the
    higher rate wins) -- pass whichever is more natural to reason about for
    a given job, or both.

    `max_attempts=1` (the default): run exactly one bounded attempt and
    return, regardless of whether the job finished -- the safe choice when
    the job's total runtime is unknown or likely exceeds
    attempt_timeout_seconds many times over (see module docstring for why:
    a single run_until_complete call is itself just one more thing that can
    be killed by whatever this environment's own duration ceiling is, so
    the outer resumption loop belongs one level up, not nested inside a
    single long-lived process). `max_attempts=None` loops internally until
    done or stalled -- only safe for jobs known to finish comfortably under
    the ceiling.

    Returns the final JobProgress read (whether or not the job completed --
    check `.remaining == 0` to tell the difference, e.g. when max_attempts
    bounded this call to fewer attempts than the job needed).
    """
    records = _read_attempt_log(progress_path)
    consecutive_stalls = records[-1].consecutive_stalls if records else 0
    last_completed = records[-1].completed if records else None

    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        attempt_started = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(command, timeout=attempt_timeout_seconds)
            if result.returncode != 0:
                logger.warning(
                    "Attempt %d exited with nonzero returncode %d (not a timeout) at %s",
                    attempts, result.returncode, attempt_started,
                )
        except subprocess.TimeoutExpired:
            logger.info(
                "Attempt %d hit its %ds bound at %s -- expected for a long job, resuming.",
                attempts, attempt_timeout_seconds, attempt_started,
            )

        progress = read_progress(progress_path)
        if progress is None:
            raise RuntimeError(
                f"Job never wrote a progress file at {progress_path} -- cannot tell whether it made any "
                "progress or safely resume it. This is a bug in the job itself (see write_progress), not "
                "something run_until_complete can recover from."
            )

        logger.info(
            "Resume attempt %d at %s: %d/%d completed, %d remaining",
            attempts, attempt_started, progress.completed, progress.total, progress.remaining,
        )

        if progress.remaining <= 0:
            logger.info("Job complete after %d attempt(s) this call.", attempts)
            return progress

        effective_min_rate = min_rate_items_per_second
        if expected_completion_seconds is not None and expected_completion_seconds > 0:
            implied = progress.total / expected_completion_seconds
            effective_min_rate = implied if effective_min_rate is None else max(effective_min_rate, implied)

        records.append(
            AttemptRecord(timestamp=attempt_started, completed=progress.completed, total=progress.total, consecutive_stalls=0)
        )
        too_slow = _is_too_slow(records, effective_min_rate, rate_window_attempts)
        no_progress = last_completed is not None and progress.completed <= last_completed
        consecutive_stalls = _update_stall_count(progress.completed, last_completed, consecutive_stalls, too_slow=too_slow)
        records[-1].consecutive_stalls = consecutive_stalls
        _append_attempt_record(progress_path, records[-1])

        if consecutive_stalls > 0:
            if too_slow:
                rate = _recent_rate(records, rate_window_attempts)
                logger.warning(
                    "Recent completion rate (%.4g items/s over the last %d attempt(s)) is below the required "
                    "minimum (%.4g items/s) (%d consecutive stall(s)).",
                    rate, min(len(records), rate_window_attempts), effective_min_rate, consecutive_stalls,
                )
            if no_progress:
                logger.warning(
                    "No forward progress since the last attempt (%d consecutive stall(s)).", consecutive_stalls
                )
            if consecutive_stalls >= max_consecutive_stalls:
                reasons = []
                if no_progress:
                    reasons.append("no forward progress")
                if too_slow:
                    reasons.append("completion rate below the required minimum")
                raise RuntimeError(
                    f"{consecutive_stalls} consecutive attempts with {' and/or '.join(reasons)} "
                    f"(stuck at {progress.completed}/{progress.total}) -- stopping rather than looping "
                    "silently forever. Investigate the job directly before resuming it again."
                )

        last_completed = progress.completed

    logger.info("Reached max_attempts=%d without completing; %d remaining.", max_attempts, progress.remaining)
    return progress


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatedly invoke a resumable job until its progress file reports zero remaining work."
    )
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--max-consecutive-stalls", type=int, default=DEFAULT_MAX_CONSECUTIVE_STALLS)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=DEFAULT_ATTEMPT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-attempts", type=int, default=1,
        help="1 (default): one bounded attempt per invocation, for an external scheduler to call repeatedly. "
        "Pass a larger number, or omit via a negative value for unbounded, only for jobs known to finish "
        "well under --attempt-timeout-seconds many times over.",
    )
    parser.add_argument(
        "--min-rate-items-per-second", type=float, default=None,
        help="Flag a job as stalled if its recent completion rate drops below this, even with nonzero progress.",
    )
    parser.add_argument(
        "--expected-completion-seconds", type=float, default=None,
        help="Convenience alternative to --min-rate-items-per-second: derives a minimum rate from "
        "progress.total / this value. Combined with --min-rate-items-per-second (if both given) by taking "
        "whichever implies the higher required rate.",
    )
    parser.add_argument("--rate-window-attempts", type=int, default=3)
    # REMAINDER must be declared last: argparse assigns everything from its
    # position on the command line onward to --command, so --command itself
    # must be the last flag on the actual invocation too.
    parser.add_argument("--command", required=True, nargs=argparse.REMAINDER, help="The job command and its args.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    max_attempts = None if args.max_attempts is not None and args.max_attempts < 0 else args.max_attempts
    run_until_complete(
        args.command,
        args.progress_file,
        max_consecutive_stalls=args.max_consecutive_stalls,
        attempt_timeout_seconds=args.attempt_timeout_seconds,
        max_attempts=max_attempts,
        min_rate_items_per_second=args.min_rate_items_per_second,
        expected_completion_seconds=args.expected_completion_seconds,
        rate_window_attempts=args.rate_window_attempts,
    )


if __name__ == "__main__":
    main()
