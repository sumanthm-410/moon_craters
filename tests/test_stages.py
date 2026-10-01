"""Tests for restartable pipeline stages.

Two behaviours carry the most weight here, because getting either wrong
produces results that look fine and are not:

1. a complete stage is **skipped** on resume, so a long reprocessing run is
   restartable at all; and
2. a stage whose config or inputs changed is **invalidated**, so a changed
   pixel scale or a re-downloaded input can never leave a stale product behind a
   completion marker that claims it is current.

Everything is offline and temp-dir only; the clock is injected where timing is
asserted, so no test sleeps.
"""
from __future__ import annotations

import json

import pytest

from crater import stages as st

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += float(seconds)


def counting_body(calls: list[str], payload: bytes = b"derived-product"):
    """A stage body that records each execution and writes every output."""

    def body(writer: st.OutputWriter, run: st.StageRun) -> None:
        calls.append(run.stage.name)
        for name in writer.staged_paths():
            with writer.open(name) as handle:
                handle.write(payload + b"\n" + name.encode())
        run.progress.advance(1)

    return body


@pytest.fixture
def workspace(tmp_path):
    """A tmp workspace with one input file, plus the paths a stage needs."""
    data = tmp_path / "data"
    data.mkdir()
    source = data / "input.txt"
    source.write_bytes(b"source bytes\n")
    return {
        "root": tmp_path,
        "input": source,
        "output": tmp_path / "out" / "product.bin",
        "markers": tmp_path / ".stages",
        "log": tmp_path / "logs" / "stage.log",
    }


def make_stage(workspace, calls, *, config=None, name="mosaic", **kwargs):
    return st.FunctionStage(
        name,
        counting_body(calls),
        inputs={"source": workspace["input"]},
        outputs={"product": workspace["output"]},
        config=config if config is not None else {"pixel_scale_m": 1.0, "k0": 1.0},
        marker_dir=workspace["markers"],
        log_path=workspace["log"],
        progress_total=1,
        progress_interval_s=0.0,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #
def test_config_fingerprint_ignores_key_order():
    a = st.fingerprint_config({"pixel_scale_m": 1.0, "lon_0": 0.0, "nested": {"b": 1, "a": 2}})
    b = st.fingerprint_config({"nested": {"a": 2, "b": 1}, "lon_0": 0.0, "pixel_scale_m": 1.0})
    assert a == b


def test_config_fingerprint_changes_with_any_value():
    base = {"pixel_scale_m": 1.0, "diameter_min_m": 20.0}
    assert st.fingerprint_config(base) != st.fingerprint_config(
        {**base, "pixel_scale_m": 1.0000001}
    )
    assert st.fingerprint_config(base) != st.fingerprint_config({**base, "extra": None})
    # 1 and 1.0 are different configs: they can reach GDAL as different things.
    assert st.fingerprint_config({"n": 1}) != st.fingerprint_config({"n": 1.0})


def test_config_fingerprint_refuses_non_finite_floats():
    with pytest.raises(st.StageConfigError, match="non-finite"):
        st.fingerprint_config({"roi_centre_lat": float("nan")})
    with pytest.raises(st.StageConfigError):
        st.fingerprint_config({"half_extent_km": float("inf")})


def test_canonical_json_is_sorted_and_compact():
    assert st.canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_input_fingerprint_tracks_content_and_size(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"one")
    first = st.fingerprint_inputs({"a": path})
    path.write_bytes(b"two")
    assert st.fingerprint_inputs({"a": path}) != first
    assert first["a"].startswith("sha256:")
    assert first["a"].endswith(":3")


def test_input_fingerprint_refuses_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        st.fingerprint_inputs({"a": tmp_path / "absent"})


def test_fingerprint_differences_name_what_changed(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"one")
    old = st.Fingerprint("s", st.fingerprint_inputs({"a": path}), st.fingerprint_config({"k": 1}))
    new = st.Fingerprint("s", old.inputs, st.fingerprint_config({"k": 2}))
    assert any("config hash changed" in d for d in new.differences(old))
    path.write_bytes(b"two")
    newer = st.Fingerprint("s", st.fingerprint_inputs({"a": path}), old.config_hash)
    assert "input 'a' changed" in newer.differences(old)


# --------------------------------------------------------------------------- #
# Execution, atomicity, markers
# --------------------------------------------------------------------------- #
def test_first_run_executes_and_writes_a_marker_after_validation(workspace):
    calls: list[str] = []
    stage = make_stage(workspace, calls)
    result = stage.run()

    assert result.action == st.ACTION_EXECUTED and result.ok
    assert calls == ["mosaic"]
    assert workspace["output"].is_file()
    assert stage.marker_path.is_file()
    marker = json.loads(stage.marker_path.read_text())
    assert marker["stage"] == "mosaic"
    assert marker["fingerprint"]["digest"] == result.fingerprint
    assert marker["outputs"]["product"]["sha256"] == st.hash_file(workspace["output"])
    assert marker["outputs"]["product"]["size_bytes"] == workspace["output"].stat().st_size
    assert stage.is_complete()


def test_no_temp_files_survive_a_successful_run(workspace):
    stage = make_stage(workspace, [])
    stage.run()
    leftovers = [p.name for p in workspace["output"].parent.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_a_raising_stage_leaves_no_output_and_no_marker(workspace):
    def exploding(writer, run):
        with writer.open("product") as handle:
            handle.write(b"half-written garbage")
        raise RuntimeError("projection failed halfway through")

    stage = st.FunctionStage(
        "boom", exploding, inputs={"source": workspace["input"]},
        outputs={"product": workspace["output"]}, config={"a": 1},
        marker_dir=workspace["markers"], log_path=workspace["log"],
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        stage.run()

    assert not workspace["output"].exists(), "a partial output must never be published"
    assert not stage.marker_path.exists()
    assert not list(workspace["output"].parent.glob("*.tmp"))
    assert "staged outputs discarded" in workspace["log"].read_text()

    # And a later, working run completes normally.
    calls: list[str] = []
    good = make_stage(workspace, calls, name="boom")
    assert good.run().action == st.ACTION_EXECUTED
    assert workspace["output"].is_file()


def test_marker_is_not_written_when_output_validation_fails(workspace):
    def writes_an_empty_file(writer, run):
        with writer.open("product") as handle:
            handle.write(b"")

    stage = st.FunctionStage(
        "empty", writes_an_empty_file, inputs={"source": workspace["input"]},
        outputs={"product": workspace["output"]}, marker_dir=workspace["markers"],
        log_path=workspace["log"],
    )
    result = stage.run()
    assert result.action == st.ACTION_FAILED
    assert "is empty" in result.reason
    assert not stage.marker_path.exists()
    # Validation ran on the staged file, so the final path was never created.
    assert not workspace["output"].exists()


def test_custom_validator_runs_against_staged_paths_before_commit(workspace):
    seen: list[str] = []

    def validator(paths):
        seen.extend(str(p) for p in paths.values())
        return False, "content check failed on purpose"

    stage = st.FunctionStage(
        "checked", counting_body([]), inputs={"source": workspace["input"]},
        outputs={"product": workspace["output"]}, validate_fn=validator,
        marker_dir=workspace["markers"], log_path=workspace["log"],
    )
    result = stage.run()
    assert result.action == st.ACTION_FAILED
    assert "content check failed on purpose" in result.reason
    assert all(path.endswith(".checked.tmp") for path in seen), seen
    assert not workspace["output"].exists()
    assert not stage.marker_path.exists()


def test_a_stage_that_writes_nothing_fails(workspace):
    stage = st.FunctionStage(
        "lazy", lambda writer, run: None, inputs={"source": workspace["input"]},
        outputs={"product": workspace["output"]}, marker_dir=workspace["markers"],
    )
    result = stage.run()
    assert result.action == st.ACTION_FAILED
    assert "did not write declared output" in result.reason
    assert not stage.marker_path.exists()


def test_writer_rejects_an_undeclared_output(workspace):
    def writes_elsewhere(writer, run):
        with writer.open("not_declared") as handle:  # pragma: no cover - raises
            handle.write(b"x")

    stage = st.FunctionStage(
        "stray", writes_elsewhere, outputs={"product": workspace["output"]},
        marker_dir=workspace["markers"],
    )
    with pytest.raises(KeyError, match="not a declared output"):
        stage.run()


def test_missing_input_fails_without_running_or_marking(workspace):
    calls: list[str] = []
    stage = st.FunctionStage(
        "needs_input", counting_body(calls),
        inputs={"source": workspace["root"] / "absent.txt"},
        outputs={"product": workspace["output"]}, marker_dir=workspace["markers"],
        log_path=workspace["log"],
    )
    result = stage.run()
    assert result.action == st.ACTION_FAILED
    assert "input(s) missing" in result.reason
    assert "absent.txt" in result.reason
    assert calls == []
    assert not stage.marker_path.exists()


# --------------------------------------------------------------------------- #
# --resume
# --------------------------------------------------------------------------- #
def test_a_complete_stage_is_skipped_on_resume(workspace):
    """The core resume case: the second run must not redo the work."""
    calls: list[str] = []
    stage = make_stage(workspace, calls)
    first = stage.run(st.RunOptions(resume=True))
    assert first.action == st.ACTION_EXECUTED

    digest_before = st.hash_file(workspace["output"])
    second = make_stage(workspace, calls).run(st.RunOptions(resume=True))

    assert second.action == st.ACTION_SKIPPED
    assert second.ok
    assert second.marker_state == st.MARKER_COMPLETE
    assert calls == ["mosaic"], "the stage body must not run twice"
    assert st.hash_file(workspace["output"]) == digest_before
    assert "already complete" in second.reason


def test_resume_false_re_executes(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    result = make_stage(workspace, calls).run(st.RunOptions(resume=False))
    assert result.action == st.ACTION_EXECUTED
    assert calls == ["mosaic", "mosaic"]


def test_force_re_executes_a_complete_stage(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    result = make_stage(workspace, calls).run(st.RunOptions(force=True))
    assert result.action == st.ACTION_EXECUTED
    assert result.invalidated
    assert result.invalidation_reasons == ("force requested",)
    assert calls == ["mosaic", "mosaic"]


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #
def test_changed_config_invalidates_the_completion_marker(workspace):
    """A changed config must re-run the stage, not silently reuse stale output."""
    calls: list[str] = []
    first = make_stage(workspace, calls, config={"pixel_scale_m": 1.0}).run()
    assert first.action == st.ACTION_EXECUTED
    marker_before = workspace["markers"] / "mosaic.done.json"
    assert marker_before.is_file()

    changed = make_stage(workspace, calls, config={"pixel_scale_m": 0.5})

    # Before running: the stage knows it is no longer complete, and says why.
    state, reason = changed.marker_state(changed.fingerprint())
    assert state == st.MARKER_STALE_CONFIG
    assert "config hash changed" in reason
    assert not changed.is_complete()

    result = changed.run(st.RunOptions(resume=True))
    assert result.action == st.ACTION_EXECUTED, "a stale stage must not be skipped"
    assert result.invalidated is True
    assert any("config hash changed" in r for r in result.invalidation_reasons)
    assert calls == ["mosaic", "mosaic"]
    assert result.fingerprint != first.fingerprint

    # The marker now describes the new fingerprint, and a third run skips again.
    marker = json.loads(marker_before.read_text())
    assert marker["fingerprint"]["digest"] == result.fingerprint
    assert make_stage(workspace, calls, config={"pixel_scale_m": 0.5}).run().action == (
        st.ACTION_SKIPPED
    )
    assert calls == ["mosaic", "mosaic"]
    assert "completion marker invalidated" in workspace["log"].read_text()


def test_changed_input_invalidates_the_completion_marker(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    workspace["input"].write_bytes(b"re-downloaded, different source bytes\n")

    stage = make_stage(workspace, calls)
    state, reason = stage.marker_state(stage.fingerprint())
    assert state == st.MARKER_STALE_INPUTS
    assert "input 'source' changed" in reason

    result = stage.run()
    assert result.action == st.ACTION_EXECUTED
    assert result.invalidated
    assert calls == ["mosaic", "mosaic"]


def test_deleted_output_invalidates_the_completion_marker(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    workspace["output"].unlink()

    stage = make_stage(workspace, calls)
    state, reason = stage.marker_state(stage.fingerprint())
    assert state == st.MARKER_OUTPUT_MISSING
    assert "is gone" in reason
    assert stage.run().action == st.ACTION_EXECUTED
    assert calls == ["mosaic", "mosaic"]


def test_modified_output_invalidates_the_completion_marker(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    # Same length, different content: a size-only check would miss this.
    data = bytearray(workspace["output"].read_bytes())
    data[0] ^= 0xFF
    workspace["output"].write_bytes(bytes(data))

    stage = make_stage(workspace, calls)
    state, reason = stage.marker_state(stage.fingerprint())
    assert state == st.MARKER_OUTPUT_CHANGED
    assert "content differs" in reason
    assert stage.run().action == st.ACTION_EXECUTED


def test_output_hash_verification_can_be_switched_off(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    data = bytearray(workspace["output"].read_bytes())
    data[0] ^= 0xFF
    workspace["output"].write_bytes(bytes(data))

    stage = make_stage(workspace, calls)
    assert stage.is_complete(verify_output_hashes=False) is True
    result = stage.run(st.RunOptions(verify_output_hashes=False))
    assert result.action == st.ACTION_SKIPPED


def test_corrupt_marker_is_treated_as_stale_not_as_complete(workspace):
    calls: list[str] = []
    stage = make_stage(workspace, calls)
    stage.run()
    stage.marker_path.write_text("{ this is not json")

    stage2 = make_stage(workspace, calls)
    state, reason = stage2.marker_state(stage2.fingerprint())
    assert state == st.MARKER_UNREADABLE
    assert "not readable JSON" in reason
    assert stage2.run().action == st.ACTION_EXECUTED


def test_marker_from_a_future_schema_is_not_trusted(workspace):
    stage = make_stage(workspace, [])
    stage.run()
    marker = json.loads(stage.marker_path.read_text())
    marker["fingerprint"]["schema_version"] = st.MARKER_SCHEMA_VERSION + 1
    stage.marker_path.write_text(json.dumps(marker))
    state, _ = stage.marker_state(stage.fingerprint())
    assert state == st.MARKER_UNREADABLE


def test_invalidate_is_idempotent(workspace):
    stage = make_stage(workspace, [])
    stage.run()
    assert stage.invalidate("manual") is True
    assert stage.invalidate("manual") is False
    assert not stage.is_complete()


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #
def test_dry_run_touches_nothing(workspace):
    calls: list[str] = []
    stage = make_stage(workspace, calls)
    result = stage.run(st.RunOptions(dry_run=True))

    assert result.action == st.ACTION_WOULD_EXECUTE
    assert result.ok
    assert "no completion marker" in result.reason
    assert calls == []
    assert not workspace["output"].exists()
    assert not stage.marker_path.exists()


def test_dry_run_reports_an_invalidation_without_performing_it(workspace):
    calls: list[str] = []
    make_stage(workspace, calls, config={"pixel_scale_m": 1.0}).run()
    stage = make_stage(workspace, calls, config={"pixel_scale_m": 0.5})

    result = stage.run(st.RunOptions(dry_run=True))
    assert result.action == st.ACTION_WOULD_EXECUTE
    assert "would invalidate" in result.reason
    assert st.MARKER_STALE_CONFIG in result.reason
    assert result.invalidated is False
    assert stage.marker_path.is_file(), "a dry run must not delete the marker"
    assert calls == ["mosaic"]


def test_dry_run_on_a_complete_stage_reports_a_skip(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    result = make_stage(workspace, calls).run(st.RunOptions(dry_run=True))
    assert result.action == st.ACTION_SKIPPED
    assert "would skip" in result.reason
    assert calls == ["mosaic"]


# --------------------------------------------------------------------------- #
# --validate-only
# --------------------------------------------------------------------------- #
def test_validate_only_passes_for_a_complete_stage(workspace):
    calls: list[str] = []
    make_stage(workspace, calls).run()
    result = make_stage(workspace, calls).run(st.RunOptions(validate_only=True))
    assert result.action == st.ACTION_VALIDATED
    assert result.ok
    assert calls == ["mosaic"]


def test_validate_only_reports_missing_outputs_without_producing_them(workspace):
    calls: list[str] = []
    stage = make_stage(workspace, calls)
    result = stage.run(st.RunOptions(validate_only=True))
    assert result.action == st.ACTION_INVALID
    assert not result.ok
    assert "product missing" in result.reason
    assert "no completion marker" in result.reason
    assert calls == []
    assert not workspace["output"].exists()


def test_validate_only_is_read_only_on_a_stale_stage(workspace):
    calls: list[str] = []
    make_stage(workspace, calls, config={"pixel_scale_m": 1.0}).run()
    stage = make_stage(workspace, calls, config={"pixel_scale_m": 0.5})
    result = stage.run(st.RunOptions(validate_only=True))
    assert result.action == st.ACTION_INVALID
    assert st.MARKER_STALE_CONFIG in result.reason
    assert stage.marker_path.is_file(), "validate-only must not mutate the marker"
    assert calls == ["mosaic"]


def test_contradictory_options_are_refused():
    with pytest.raises(ValueError, match="pass one"):
        st.RunOptions(dry_run=True, validate_only=True)
    with pytest.raises(ValueError, match="force"):
        st.RunOptions(force=True, validate_only=True)


# --------------------------------------------------------------------------- #
# Progress and logging
# --------------------------------------------------------------------------- #
def test_eta_is_computed_from_observed_throughput():
    clock = FakeClock()
    reporter = st.ProgressReporter(total=100, unit="tiles", clock=clock, min_interval_s=0.0)
    reporter.start()
    clock.tick(10.0)
    reporter.advance(25)

    snapshot = reporter.snapshot()
    assert snapshot.rate_per_s == pytest.approx(2.5)
    # 75 tiles left at 2.5 tiles/s, measured, not guessed.
    assert snapshot.eta_s == pytest.approx(30.0)
    assert snapshot.fraction == pytest.approx(0.25)
    assert "eta 30.0s (from observed throughput)" in snapshot.message()


def test_no_eta_without_a_declared_total():
    clock = FakeClock()
    reporter = st.ProgressReporter(total=None, unit="products", clock=clock)
    reporter.start()
    clock.tick(5.0)
    reporter.advance(3)

    snapshot = reporter.snapshot()
    assert snapshot.eta_s is None
    assert snapshot.eta_reason == st.ETA_NO_TOTAL
    assert snapshot.fraction is None
    message = snapshot.message()
    assert "3 products done" in message
    assert "no eta" in message
    assert "remaining" not in message


def test_no_eta_before_any_throughput_is_observed():
    clock = FakeClock()
    reporter = st.ProgressReporter(total=10, clock=clock)
    reporter.start()
    clock.tick(30.0)  # time passed, nothing finished
    snapshot = reporter.snapshot()
    assert snapshot.eta_s is None
    assert snapshot.rate_per_s is None
    assert snapshot.eta_reason == st.ETA_NO_THROUGHPUT


def test_completed_progress_reports_no_eta():
    clock = FakeClock()
    reporter = st.ProgressReporter(total=2, clock=clock)
    reporter.start()
    clock.tick(4.0)
    reporter.advance(2)
    snapshot = reporter.snapshot()
    assert snapshot.eta_s is None
    assert snapshot.eta_reason == st.ETA_DONE


def test_heartbeat_is_rate_limited_but_always_brackets_the_run():
    clock = FakeClock()
    lines: list[str] = []
    reporter = st.ProgressReporter(
        total=4, sink=lines.append, clock=clock, min_interval_s=10.0, label="tiling",
    )
    reporter.start()
    for _ in range(4):
        clock.tick(1.0)
        reporter.advance(1)
    reporter.finish()
    assert len(lines) == 2, "one line at start, one at finish; the rest rate-limited"
    assert all(line.startswith("tiling: ") for line in lines)
    assert "4/4" in lines[-1]


def test_progress_reaches_the_stage_log(workspace):
    stage = make_stage(workspace, [])
    stage.run()
    log = workspace["log"].read_text()
    assert "[mosaic]" in log
    assert "1/1 items" in log
    assert "completed in" in log


def test_stage_log_redacts_credentials(workspace):
    stage = make_stage(workspace, [])
    stage.log(
        "fetching https://pds.lroc.asu.edu/x.IMG"
        "?X-Amz-Signature=5d41402abc4b2a76b9719d911017c592"
    )
    log = workspace["log"].read_text()
    assert "5d41402abc4b2a76b9719d911017c592" not in log
    assert "REDACTED" in log
    assert "pds.lroc.asu.edu/x.IMG" in log


def test_progress_total_must_be_sane():
    with pytest.raises(ValueError, match="non-negative"):
        st.ProgressReporter(total=-1)


def test_stage_name_must_be_path_safe():
    with pytest.raises(ValueError, match="path-safe"):
        st.Stage("a/b")
    with pytest.raises(ValueError):
        st.Stage("")


def test_base_stage_has_no_default_body(workspace):
    stage = st.Stage("abstract", outputs={"product": workspace["output"]},
                     marker_dir=workspace["markers"])
    with pytest.raises(NotImplementedError, match="abstract"):
        stage.run()


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def test_pipeline_stops_at_the_first_failure(workspace):
    calls: list[str] = []
    first = st.FunctionStage(
        "one", counting_body(calls), inputs={"source": workspace["input"]},
        outputs={"product": workspace["root"] / "out" / "one.bin"},
        marker_dir=workspace["markers"],
    )
    blocked = st.FunctionStage(
        "two", counting_body(calls), inputs={"missing": workspace["root"] / "absent"},
        outputs={"product": workspace["root"] / "out" / "two.bin"},
        marker_dir=workspace["markers"],
    )
    never = st.FunctionStage(
        "three", counting_body(calls), outputs={"product": workspace["root"] / "out" / "3.bin"},
        marker_dir=workspace["markers"],
    )
    results = st.run_pipeline([first, blocked, never])
    assert [r.action for r in results] == [st.ACTION_EXECUTED, st.ACTION_FAILED]
    assert calls == ["one"], "a downstream stage must not run on a failed upstream"


def test_pipeline_is_restartable_end_to_end(workspace):
    """Run a two-stage pipeline twice; the second run does no work."""
    calls: list[str] = []
    stage_one_out = workspace["root"] / "out" / "one.bin"

    def build(config_two):
        one = st.FunctionStage(
            "one", counting_body(calls), inputs={"source": workspace["input"]},
            outputs={"product": stage_one_out}, config={"a": 1},
            marker_dir=workspace["markers"],
        )
        two = st.FunctionStage(
            "two", counting_body(calls), inputs={"upstream": stage_one_out},
            outputs={"product": workspace["root"] / "out" / "two.bin"},
            config=config_two, marker_dir=workspace["markers"],
        )
        return [one, two]

    assert [r.action for r in st.run_pipeline(build({"b": 1}))] == [
        st.ACTION_EXECUTED, st.ACTION_EXECUTED
    ]
    assert calls == ["one", "two"]

    assert [r.action for r in st.run_pipeline(build({"b": 1}))] == [
        st.ACTION_SKIPPED, st.ACTION_SKIPPED
    ]
    assert calls == ["one", "two"]

    # Changing only the second stage's config re-runs only the second stage.
    results = st.run_pipeline(build({"b": 2}))
    assert [r.action for r in results] == [st.ACTION_SKIPPED, st.ACTION_EXECUTED]
    assert calls == ["one", "two", "two"]
    assert results[1].invalidated


def test_an_upstream_rerun_invalidates_the_downstream_stage(workspace):
    """The whole point of input fingerprinting: staleness propagates."""
    calls: list[str] = []
    stage_one_out = workspace["root"] / "out" / "one.bin"
    counter = {"n": 0}

    def changing_body(writer, run):
        calls.append("one")
        counter["n"] += 1
        with writer.open("product") as handle:
            handle.write(f"revision {counter['n']}\n".encode())

    def one(config):
        return st.FunctionStage(
            "one", changing_body, inputs={"source": workspace["input"]},
            outputs={"product": stage_one_out}, config=config,
            marker_dir=workspace["markers"],
        )

    def two():
        return st.FunctionStage(
            "two", counting_body(calls), inputs={"upstream": stage_one_out},
            outputs={"product": workspace["root"] / "out" / "two.bin"},
            config={"b": 1}, marker_dir=workspace["markers"],
        )

    assert one({"a": 1}).run().action == st.ACTION_EXECUTED
    assert two().run().action == st.ACTION_EXECUTED
    # Re-run the first stage with a different config: its output changes...
    assert one({"a": 2}).run().action == st.ACTION_EXECUTED
    # ...so the second stage's input fingerprint no longer matches its marker.
    downstream = two()
    state, reason = downstream.marker_state(downstream.fingerprint())
    assert state == st.MARKER_STALE_INPUTS
    assert "input 'upstream' changed" in reason
    result = downstream.run()
    assert result.action == st.ACTION_EXECUTED
    assert result.invalidated
