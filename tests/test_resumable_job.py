import json
import sys
import textwrap
from pathlib import Path

import pytest

from src.resumable_job import (
    AttemptRecord,
    JobProgress,
    _attempt_log_path,
    _is_too_slow,
    _read_attempt_log,
    _recent_rate,
    _update_stall_count,
    read_progress,
    run_until_complete,
    write_progress,
)


# ---------- write_progress / read_progress ----------


def test_write_progress_then_read_progress_round_trips(tmp_path):
    path = tmp_path / "progress.json"
    write_progress(path, total=10, completed=4, extra={"note": "hi"})
    progress = read_progress(path)
    assert progress.total == 10
    assert progress.completed == 4
    assert progress.remaining == 6
    assert progress.extra == {"note": "hi"}


def test_write_progress_clamps_remaining_at_zero_if_completed_exceeds_total(tmp_path):
    path = tmp_path / "progress.json"
    write_progress(path, total=5, completed=7)
    assert read_progress(path).remaining == 0


def test_read_progress_returns_none_for_missing_file(tmp_path):
    assert read_progress(tmp_path / "does_not_exist.json") is None


def test_write_progress_leaves_no_leftover_temp_file(tmp_path):
    path = tmp_path / "progress.json"
    write_progress(path, total=10, completed=4)
    assert not path.with_name(path.name + ".tmp").exists()
    assert path.exists()


# ---------- _read_attempt_log: corrupt/truncated line handling ----------


def _valid_line(completed: int, total: int = 100, consecutive_stalls: int = 0) -> str:
    return json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "completed": completed,
            "total": total,
            "consecutive_stalls": consecutive_stalls,
        }
    )


def test_read_attempt_log_tolerates_a_truncated_trailing_line(tmp_path, caplog):
    progress_path = tmp_path / "progress.json"
    log_path = _attempt_log_path(progress_path)
    log_path.write_text(_valid_line(5) + "\n" + '{"timestamp": "2026-01-01T00:10:00+00:00", "completed": 8, "tot')

    with caplog.at_level("WARNING"):
        records = _read_attempt_log(progress_path)

    assert [r.completed for r in records] == [5]
    assert any("truncated/corrupt" in message for message in caplog.messages)


def test_read_attempt_log_tolerates_a_trailing_blank_line_as_not_corrupt(tmp_path):
    # A trailing newline after the last valid record is normal file
    # formatting, not corruption -- must not be flagged or dropped.
    progress_path = tmp_path / "progress.json"
    log_path = _attempt_log_path(progress_path)
    log_path.write_text(_valid_line(5) + "\n" + _valid_line(9) + "\n\n")

    records = _read_attempt_log(progress_path)
    assert [r.completed for r in records] == [5, 9]


def test_read_attempt_log_raises_on_a_corrupt_non_trailing_line(tmp_path):
    progress_path = tmp_path / "progress.json"
    log_path = _attempt_log_path(progress_path)
    log_path.write_text(_valid_line(5) + "\n" + "{not even close to json" + "\n" + _valid_line(9) + "\n")

    with pytest.raises(RuntimeError, match="NOT the trailing line"):
        _read_attempt_log(progress_path)


def test_read_attempt_log_returns_empty_list_when_log_does_not_exist(tmp_path):
    progress_path = tmp_path / "progress.json"
    assert _read_attempt_log(progress_path) == []


# ---------- _update_stall_count (pure logic) ----------


def test_update_stall_count_first_attempt_never_stalls():
    assert _update_stall_count(completed=0, last_completed=None, consecutive_stalls=0) == 0


def test_update_stall_count_increments_when_no_progress():
    assert _update_stall_count(completed=5, last_completed=5, consecutive_stalls=2) == 3


def test_update_stall_count_resets_when_progress_resumes():
    assert _update_stall_count(completed=8, last_completed=5, consecutive_stalls=3) == 0


def test_update_stall_count_counts_a_regression_as_a_stall_too():
    # completed going backward shouldn't happen for a well-behaved job, but
    # treat it the same as "no forward progress" rather than a crash.
    assert _update_stall_count(completed=3, last_completed=5, consecutive_stalls=1) == 2


def test_update_stall_count_too_slow_counts_as_a_stall_even_with_forward_progress():
    # completed(6) > last_completed(5) -- genuine forward progress -- but
    # too_slow=True still counts it against the stall counter, since a
    # too-slow job is surfaced the same way a stopped one is.
    assert _update_stall_count(completed=6, last_completed=5, consecutive_stalls=0, too_slow=True) == 1


def test_update_stall_count_too_slow_can_flag_even_the_first_attempt():
    # Unlike the no-progress check, a rate judgement doesn't need a
    # *previous* attempt to compare completed against -- only enough
    # history in the window to compute a rate, which _recent_rate already
    # guards (returns None with <2 records) independent of this function.
    assert _update_stall_count(completed=1, last_completed=None, consecutive_stalls=0, too_slow=True) == 1


# ---------- _recent_rate / _is_too_slow ----------


def _record(timestamp: str, completed: int, total: int = 100) -> AttemptRecord:
    return AttemptRecord(timestamp=timestamp, completed=completed, total=total, consecutive_stalls=0)


def test_recent_rate_returns_none_with_fewer_than_two_records():
    assert _recent_rate([_record("2026-01-01T00:00:00+00:00", 5)], window=3) is None
    assert _recent_rate([], window=3) is None


def test_recent_rate_computes_items_per_second_between_first_and_last_in_window():
    records = [
        _record("2026-01-01T00:00:00+00:00", 0),
        _record("2026-01-01T00:00:10+00:00", 20),
    ]
    assert _recent_rate(records, window=3) == pytest.approx(2.0)


def test_recent_rate_uses_only_the_most_recent_window_not_all_history():
    # Fast start (0 -> 100 over 10s = 10/s), then a much slower recent
    # stretch (100 -> 105 over 10s = 0.5/s) -- windowed to the last 2
    # records should reflect the *recent* slow rate, not the all-time
    # average (which would still look fast, exactly the failure mode this
    # module is designed around: a job that used to be fast but has since
    # decelerated).
    records = [
        _record("2026-01-01T00:00:00+00:00", 0),
        _record("2026-01-01T00:00:10+00:00", 100),
        _record("2026-01-01T00:00:20+00:00", 105),
    ]
    assert _recent_rate(records, window=2) == pytest.approx(0.5)
    all_time_rate = _recent_rate(records, window=10)
    assert all_time_rate > 5.0  # the diluted all-time average, for contrast


def test_is_too_slow_false_when_no_min_rate_given():
    records = [_record("2026-01-01T00:00:00+00:00", 0), _record("2026-01-01T00:00:10+00:00", 1)]
    assert _is_too_slow(records, min_rate=None, window=3) is False


def test_is_too_slow_true_when_recent_rate_below_minimum():
    records = [_record("2026-01-01T00:00:00+00:00", 0), _record("2026-01-01T00:00:10+00:00", 1)]  # 0.1 items/s
    assert _is_too_slow(records, min_rate=1.0, window=3) is True


def test_is_too_slow_false_when_recent_rate_meets_minimum():
    records = [_record("2026-01-01T00:00:00+00:00", 0), _record("2026-01-01T00:00:10+00:00", 100)]  # 10 items/s
    assert _is_too_slow(records, min_rate=1.0, window=3) is False


# ---------- run_until_complete (real subprocess integration) ----------


_FAKE_JOB_SOURCE = textwrap.dedent(
    """
    import argparse, json, sys, time
    from pathlib import Path
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--advance", type=int, default=1)
    parser.add_argument("--advance-schedule", type=str, default=None,
                         help="Comma-separated per-call advance amounts; last value repeats once exhausted.")
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--call-count-file", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    call_index = 0
    if args.call_count_file is not None:
        if args.call_count_file.exists():
            call_index = int(args.call_count_file.read_text())
        args.call_count_file.write_text(str(call_index + 1))

    if args.advance_schedule is not None:
        schedule = [int(v) for v in args.advance_schedule.split(",")]
        advance = schedule[min(call_index, len(schedule) - 1)]
    else:
        advance = args.advance

    completed = 0
    if args.progress_file.exists():
        completed = json.loads(args.progress_file.read_text())["completed"]
    completed = min(completed + advance, args.total)
    args.progress_file.write_text(json.dumps({
        "total": args.total, "completed": completed, "remaining": max(args.total - completed, 0),
        "updated_at": datetime.now(timezone.utc).isoformat(), "extra": None,
    }))

    time.sleep(args.sleep)
    sys.exit(args.exit_code)
    """
)


@pytest.fixture
def fake_job(tmp_path) -> Path:
    script = tmp_path / "fake_job.py"
    script.write_text(_FAKE_JOB_SOURCE)
    return script


def _command(fake_job: Path, progress_path: Path, total: int, advance: int = 1, sleep: float = 0.0, exit_code: int = 0) -> list[str]:
    return [
        sys.executable, str(fake_job),
        "--total", str(total), "--advance", str(advance),
        "--progress-file", str(progress_path), "--sleep", str(sleep), "--exit-code", str(exit_code),
    ]


def test_run_until_complete_finishes_in_one_attempt_when_the_job_finishes_the_work(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=3, advance=3)
    result = run_until_complete(command, progress_path, max_attempts=1)
    assert result.remaining == 0
    assert result.completed == 3


def test_run_until_complete_loops_internally_across_multiple_attempts_until_done(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=9, advance=3)
    result = run_until_complete(command, progress_path, max_attempts=None)
    assert result.remaining == 0
    assert result.completed == 9


def test_run_until_complete_bounded_by_max_attempts_returns_partial_progress(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=9, advance=3)
    result = run_until_complete(command, progress_path, max_attempts=1)
    assert result.completed == 3
    assert result.remaining == 6


def test_run_until_complete_treats_a_timed_out_attempt_as_expected_not_an_error(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    # Writes progress *before* sleeping -- the timeout kill happens during
    # the sleep, after the write, mirroring the real incident this module
    # is modeled on (progress persisted, then the process got killed).
    command = _command(fake_job, progress_path, total=9, advance=3, sleep=5.0)
    result = run_until_complete(command, progress_path, attempt_timeout_seconds=1, max_attempts=1)
    assert result.completed == 3
    assert result.remaining == 6


def test_run_until_complete_raises_if_progress_file_never_written(tmp_path):
    progress_path = tmp_path / "progress.json"
    command = [sys.executable, "-c", "import sys; sys.exit(0)"]
    with pytest.raises(RuntimeError, match="never wrote a progress file"):
        run_until_complete(command, progress_path, max_attempts=1)


def test_run_until_complete_raises_after_max_consecutive_stalls_with_no_progress(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=5, advance=0)
    with pytest.raises(RuntimeError, match="consecutive attempts with no forward progress"):
        run_until_complete(command, progress_path, max_consecutive_stalls=3, max_attempts=None)


def test_run_until_complete_logs_nonzero_exit_as_warning_but_still_makes_progress(tmp_path, fake_job, caplog):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=6, advance=3, exit_code=1)
    with caplog.at_level("WARNING"):
        result = run_until_complete(command, progress_path, max_attempts=None)
    assert result.remaining == 0
    assert any("nonzero returncode" in message for message in caplog.messages)


def test_run_until_complete_resets_stall_counter_when_progress_resumes(tmp_path, fake_job):
    # A job that stalls on its first two calls, then genuinely advances on
    # the third (all within one run_until_complete call), must not raise --
    # forward progress on the third attempt has to reset the stall counter
    # rather than getting evaluated against the earlier stalled attempts.
    # max_consecutive_stalls=2 means a *third* consecutive stall would have
    # raised -- it doesn't, because the third call actually advances.
    progress_path = tmp_path / "progress.json"
    call_count_path = tmp_path / "call_count.txt"
    command = _command(fake_job, progress_path, total=9, advance=0)
    command += ["--advance-schedule", "0,0,9", "--call-count-file", str(call_count_path)]

    result = run_until_complete(command, progress_path, max_consecutive_stalls=2, max_attempts=None)
    assert result.remaining == 0


# ---------- run_until_complete: too-slow-but-nonzero-progress detection ----------


def test_run_until_complete_raises_for_nonzero_progress_below_the_minimum_rate(tmp_path, fake_job):
    # Advances by 1 item every ~0.3s (~3.3 items/s) -- genuine, nonzero
    # forward progress every single attempt, which _update_stall_count's
    # original no-progress check would never flag. A far higher required
    # rate must still catch it.
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=1000, advance=1, sleep=0.3)
    with pytest.raises(RuntimeError, match="consecutive attempts with completion rate below the required minimum"):
        run_until_complete(
            command, progress_path, max_consecutive_stalls=3, max_attempts=None,
            min_rate_items_per_second=100.0, rate_window_attempts=2,
        )


def test_run_until_complete_expected_completion_seconds_derives_a_minimum_rate(tmp_path, fake_job):
    # total=1000 with expected_completion_seconds=1 implies a required rate
    # of 1000 items/s -- the fake job's real ~3.3 items/s can't come close.
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=1000, advance=1, sleep=0.3)
    with pytest.raises(RuntimeError, match="completion rate below the required minimum"):
        run_until_complete(
            command, progress_path, max_consecutive_stalls=3, max_attempts=None,
            expected_completion_seconds=1.0, rate_window_attempts=2,
        )


def test_run_until_complete_does_not_flag_a_job_meeting_the_minimum_rate(tmp_path, fake_job):
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=6, advance=3)
    result = run_until_complete(
        command, progress_path, max_attempts=None, min_rate_items_per_second=0.001,
    )
    assert result.remaining == 0


# ---------- run_until_complete: stall state persists across separate calls ----------


def test_run_until_complete_stall_count_persists_across_separate_calls(tmp_path, fake_job):
    # The recommended real-world usage pattern is max_attempts=1 with an
    # external scheduler invoking run_until_complete again as a fresh
    # process each time -- stall detection has to accumulate across those
    # separate calls (via the on-disk attempt log), not reset to zero every
    # time just because it's a new Python process with empty local
    # variables. A job that never advances must still get flagged within
    # max_consecutive_stalls *calls*, not just within one long-lived call.
    progress_path = tmp_path / "progress.json"
    command = _command(fake_job, progress_path, total=5, advance=0)

    run_until_complete(command, progress_path, max_consecutive_stalls=2, max_attempts=1)  # attempt 1: baseline
    run_until_complete(command, progress_path, max_consecutive_stalls=2, max_attempts=1)  # attempt 2: 1 stall
    with pytest.raises(RuntimeError, match="consecutive attempts"):
        run_until_complete(command, progress_path, max_consecutive_stalls=2, max_attempts=1)  # attempt 3: 2 stalls -> raise


def test_run_until_complete_survives_a_truncated_attempt_log_from_a_kill_mid_write(tmp_path, fake_job):
    # Simulates exactly the scenario a kill at the exact moment of an
    # _append_attempt_record write would produce: a valid record from an
    # earlier, fully-completed attempt, followed by a truncated trailing
    # line from the attempt that was in progress when the process died.
    # Resuming must not crash -- it should treat the truncated attempt as
    # if it never happened and continue from the last valid record.
    progress_path = tmp_path / "progress.json"
    write_progress(progress_path, total=9, completed=3)
    log_path = _attempt_log_path(progress_path)
    log_path.write_text(
        _valid_line(3, total=9) + "\n" + '{"timestamp": "2026-01-01T00:10:00+00:00", "completed": 6, "tot'
    )

    command = _command(fake_job, progress_path, total=9, advance=3)
    result = run_until_complete(command, progress_path, max_attempts=1)
    assert result.completed == 6
    assert result.remaining == 3
