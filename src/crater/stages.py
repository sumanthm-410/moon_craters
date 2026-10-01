"""Restartable pipeline stages with fingerprinted completion markers.

A survey of this kind is run many times, in pieces, over days, on a machine
with 30 GiB of disk (config/project.yaml).  Stages must therefore be
*restartable*: cheap to skip when already done, and -- the part that is usually
got wrong -- **not** skippable when the thing they were done *for* has changed.

The contract
------------
``Stage.run(RunOptions(...))`` always returns a :class:`StageResult` whose
``action`` is one of :data:`ACTION_EXECUTED`, :data:`ACTION_SKIPPED`,
:data:`ACTION_WOULD_EXECUTE`, :data:`ACTION_VALIDATED`, :data:`ACTION_INVALID`
or :data:`ACTION_FAILED`.

* ``--dry-run`` (``RunOptions(dry_run=True)``) touches nothing on disk: no
  outputs, no marker, no invalidation.  It reports what *would* happen and why.
* ``--resume`` (the default) skips a stage whose completion marker is present
  **and** whose fingerprint still matches **and** whose recorded outputs are
  still on disk with the recorded digests.
* ``--validate-only`` re-checks existing outputs and reports, without executing
  and without mutating any marker.

Fingerprinting and invalidation
-------------------------------
A marker records a hash over: the stage name, the digest and size of every
declared input, and a canonical hash of the stage's relevant config.  On the
next run the fingerprint is recomputed; if any part differs the marker is
**invalidated** (deleted) and the stage re-executes.  Without this, changing
``pixel_scale_m`` or the ROI centre would leave every downstream product
silently stale while every stage reported "already done" -- the single most
expensive failure mode in a long reprocessing pipeline.

Atomicity
---------
Outputs are written to sibling temp files, fsynced, validated *in place*, and
only then renamed over their final paths; the completion marker is written only
after that.  So at no point does a reader see a committed output with a
completion marker that does not describe it, and a crash leaves either the
previous state or the new one.

Honest progress
---------------
:class:`ProgressReporter` reports an ETA **only** when a total is declared and
a throughput has actually been observed; otherwise it reports counts and says
why there is no estimate.  A fabricated "2 minutes remaining" is a lie the
pipeline tells about its own measurements, and this project does not tell it
(INTERFACES.md rule 6).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .download import redact_text

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Bumped when the marker file layout changes; an older marker is then treated
#: as unreadable (so stale, so re-run) rather than misinterpreted.
MARKER_SCHEMA_VERSION = 1

#: Suffix of a completion marker, written inside the stage's marker directory.
MARKER_SUFFIX = ".done.json"

ACTION_EXECUTED = "executed"
ACTION_SKIPPED = "skipped"
ACTION_WOULD_EXECUTE = "would_execute"
ACTION_VALIDATED = "validated"
ACTION_INVALID = "invalid"
ACTION_FAILED = "failed"

MARKER_COMPLETE = "complete"
MARKER_MISSING = "missing"
MARKER_UNREADABLE = "unreadable"
MARKER_STALE_CONFIG = "stale_config"
MARKER_STALE_INPUTS = "stale_inputs"
MARKER_STALE_STAGE = "stale_stage_definition"
MARKER_OUTPUT_MISSING = "output_missing"
MARKER_OUTPUT_CHANGED = "output_changed"

#: Marker states that mean "there is a marker, but it no longer applies".
STALE_MARKER_STATES: frozenset[str] = frozenset({
    MARKER_UNREADABLE, MARKER_STALE_CONFIG, MARKER_STALE_INPUTS,
    MARKER_STALE_STAGE, MARKER_OUTPUT_MISSING, MARKER_OUTPUT_CHANGED,
})

_HASH_CHUNK = 1 << 20


class StageError(Exception):
    """Base class for stage problems."""


class StageConfigError(StageError):
    """The stage's declared config cannot be fingerprinted reproducibly."""


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #
def hash_file(path: str | os.PathLike[str], algorithm: str = "sha256") -> str:
    """Streamed hex digest of a file."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    """Convert ``value`` to something ``json.dumps`` can order reproducibly."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        # NaN/inf would serialise as JavaScript literals that are not JSON and
        # that compare unequal to themselves -- a fingerprint built on them
        # could never match.  Refuse instead of producing a useless hash.
        if value != value or value in (float("inf"), float("-inf")):
            raise StageConfigError(
                f"config contains a non-finite float ({value!r}); a stage "
                f"fingerprint must be reproducible, so this is refused"
            )
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(v) for v in value), key=repr)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    # numpy scalars and anything else with a sane repr: use the repr so the
    # fingerprint is at least stable within a run, and visible in the marker.
    return {"__repr__": repr(value)}


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True,
    )


def fingerprint_config(config: Mapping[str, Any] | None) -> str:
    """``sha256`` over the canonical JSON of a config mapping.

    Key order is irrelevant; value *types* are not (``1`` and ``1.0``
    fingerprint differently, because ``pixel_scale_m: 1`` and
    ``pixel_scale_m: 1.0`` can reach GDAL as different things).
    """
    return hashlib.sha256(canonical_json(config or {}).encode("utf-8")).hexdigest()


def fingerprint_inputs(inputs: Mapping[str, str | os.PathLike[str]]) -> dict[str, str]:
    """Digest and size of every declared input, keyed by logical name.

    Raises ``FileNotFoundError`` if an input is missing: a fingerprint over
    absent inputs would be a statement we cannot support.
    """
    out: dict[str, str] = {}
    for name, path in sorted(inputs.items()):
        p = Path(path)
        stat = p.stat()  # raises FileNotFoundError, deliberately
        out[name] = f"sha256:{hash_file(p)}:{stat.st_size}"
    return out


@dataclass(frozen=True)
class Fingerprint:
    """The identity of a stage execution: stage name, inputs, config."""

    stage: str
    inputs: dict[str, str]
    config_hash: str
    schema_version: int = MARKER_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "schema_version": self.schema_version,
                    "stage": self.stage,
                    "inputs": self.inputs,
                    "config_hash": self.config_hash,
                }
            ).encode("utf-8")
        ).hexdigest()

    def differences(self, other: "Fingerprint | None") -> list[str]:
        """Human-readable list of what changed relative to ``other``."""
        if other is None:
            return ["no previous fingerprint"]
        diffs: list[str] = []
        if self.schema_version != other.schema_version:
            diffs.append(
                f"marker schema {other.schema_version} != {self.schema_version}"
            )
        if self.stage != other.stage:
            diffs.append(f"stage name {other.stage!r} != {self.stage!r}")
        if self.config_hash != other.config_hash:
            diffs.append(
                f"config hash changed ({other.config_hash[:12]} -> {self.config_hash[:12]})"
            )
        for name in sorted(set(self.inputs) | set(other.inputs)):
            mine, theirs = self.inputs.get(name), other.inputs.get(name)
            if mine != theirs:
                if theirs is None:
                    diffs.append(f"input {name!r} is new")
                elif mine is None:
                    diffs.append(f"input {name!r} was removed")
                else:
                    diffs.append(f"input {name!r} changed")
        return diffs

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "inputs": dict(self.inputs),
            "config_hash": self.config_hash,
            "digest": self.digest,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Fingerprint":
        return cls(
            stage=str(data["stage"]),
            inputs={str(k): str(v) for k, v in dict(data.get("inputs") or {}).items()},
            config_hash=str(data["config_hash"]),
            schema_version=int(data.get("schema_version", 0)),
        )


# --------------------------------------------------------------------------- #
# Honest progress reporting
# --------------------------------------------------------------------------- #
ETA_NO_TOTAL = "no total declared, so no ETA can be computed"
ETA_NO_THROUGHPUT = "no throughput observed yet, so no ETA can be computed"
ETA_DONE = "complete"


@dataclass(frozen=True)
class Progress:
    """A progress snapshot.  ``eta_s`` is ``None`` unless honestly computable."""

    done: int
    total: int | None
    elapsed_s: float
    rate_per_s: float | None
    eta_s: float | None
    eta_reason: str
    unit: str = "items"

    @property
    def fraction(self) -> float | None:
        if self.total in (None, 0):
            return None
        return self.done / float(self.total)  # type: ignore[operator]

    def message(self) -> str:
        """One-line progress text.  Never contains an invented estimate."""
        if self.total is None:
            head = f"{self.done} {self.unit} done"
        else:
            head = f"{self.done}/{self.total} {self.unit}"
        parts = [head, f"{self.elapsed_s:.1f}s elapsed"]
        if self.rate_per_s is not None:
            parts.append(f"{self.rate_per_s:.2f} {self.unit}/s")
        if self.eta_s is not None:
            parts.append(f"eta {self.eta_s:.1f}s (from observed throughput)")
        else:
            parts.append(f"no eta: {self.eta_reason}")
        return ", ".join(parts)


class ProgressReporter:
    """Heartbeat/progress hook for a long stage.

    ``sink`` receives a formatted line at most every ``min_interval_s`` seconds
    of ``clock`` time, plus one at ``start`` and one at ``finish`` so a command
    always shows something immediately and a final count at the end.
    """

    def __init__(
        self,
        total: int | None = None,
        *,
        unit: str = "items",
        label: str = "",
        sink: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        min_interval_s: float = 5.0,
    ) -> None:
        if total is not None and total < 0:
            raise ValueError("total must be non-negative or None")
        self.total = total
        self.unit = unit
        self.label = label
        self.sink = sink
        self.clock = clock
        self.min_interval_s = float(min_interval_s)
        self.done = 0
        self._t0: float | None = None
        self._last_emit: float | None = None

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        self._t0 = self.clock()
        self._last_emit = None
        self._emit(force=True)

    def advance(self, n: int = 1) -> None:
        if self._t0 is None:
            self.start()
        self.done += int(n)
        self._emit()

    def finish(self) -> None:
        if self._t0 is None:
            self.start()
        self._emit(force=True)

    # -- reporting ------------------------------------------------------- #
    def snapshot(self) -> Progress:
        """Current progress, with an ETA only when one is honestly available."""
        t0 = self._t0 if self._t0 is not None else self.clock()
        elapsed = max(0.0, self.clock() - t0)
        rate = self.done / elapsed if (elapsed > 0.0 and self.done > 0) else None
        eta: float | None = None
        if self.total is None:
            reason = ETA_NO_TOTAL
        elif self.done >= self.total:
            reason = ETA_DONE
        elif rate is None:
            # Either nothing finished yet or no time has passed: there is no
            # measured throughput, so there is no estimate to give.
            reason = ETA_NO_THROUGHPUT
        else:
            eta = (self.total - self.done) / rate
            reason = f"from {self.done} observed {self.unit} over {elapsed:.1f}s"
        return Progress(
            done=self.done, total=self.total, elapsed_s=elapsed, rate_per_s=rate,
            eta_s=eta, eta_reason=reason, unit=self.unit,
        )

    def _emit(self, force: bool = False) -> None:
        if self.sink is None:
            return
        now = self.clock()
        if not force and self._last_emit is not None and now - self._last_emit < self.min_interval_s:
            return
        self._last_emit = now
        prefix = f"{self.label}: " if self.label else ""
        self.sink(prefix + self.snapshot().message())


# --------------------------------------------------------------------------- #
# Run options and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunOptions:
    """The ``--dry-run`` / ``--resume`` / ``--validate-only`` triple.

    ``resume`` defaults to true because skipping completed work is the safe
    behaviour *given* that the fingerprint check is what decides completion.
    ``force`` re-executes even a valid, complete stage.
    """

    dry_run: bool = False
    resume: bool = True
    validate_only: bool = False
    force: bool = False
    verify_output_hashes: bool = True

    def __post_init__(self) -> None:
        if self.dry_run and self.validate_only:
            raise ValueError(
                "dry_run and validate_only are different questions "
                "('what would you do' vs 'are the existing outputs good'); "
                "pass one"
            )
        if self.force and self.validate_only:
            raise ValueError("force cannot be combined with validate_only")


@dataclass(frozen=True)
class StageResult:
    """What a stage run did, and why."""

    stage: str
    action: str
    ok: bool
    reason: str
    marker_state: str
    fingerprint: str
    outputs: dict[str, Path] = field(default_factory=dict)
    invalidated: bool = False
    invalidation_reasons: tuple[str, ...] = ()
    duration_s: float | None = None
    progress: Progress | None = None
    log_path: Path | None = None

    @property
    def executed(self) -> bool:
        return self.action == ACTION_EXECUTED

    @property
    def skipped(self) -> bool:
        return self.action == ACTION_SKIPPED

    def describe(self) -> str:
        text = f"{self.stage}: {self.action} ({self.reason})"
        if self.invalidated:
            text += " [marker invalidated: " + "; ".join(self.invalidation_reasons) + "]"
        return text


# --------------------------------------------------------------------------- #
# Atomic output writing
# --------------------------------------------------------------------------- #
class OutputWriter:
    """Writes a stage's declared outputs atomically.

    Each output gets a sibling temp path.  The stage writes to the temp paths
    (``writer.path(name)`` or ``writer.open(name)``); :meth:`commit` fsyncs and
    renames them into place.  :meth:`discard` removes them.  Validation runs
    against the temp paths *before* commit, which is what makes "marker only
    after outputs validate" true even if the rename itself were to fail.
    """

    def __init__(self, stage: str, outputs: Mapping[str, str | os.PathLike[str]]) -> None:
        self.stage = stage
        self.final: dict[str, Path] = {name: Path(p) for name, p in outputs.items()}
        self.staged: dict[str, Path] = {
            name: p.with_name(f"{p.name}.{stage}.tmp") for name, p in self.final.items()
        }
        self._committed = False
        for p in self.final.values():
            p.parent.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """The temp path to write output ``name`` to."""
        try:
            return self.staged[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a declared output of stage {self.stage!r}; "
                f"declared: {sorted(self.final)}"
            ) from None

    def final_path(self, name: str) -> Path:
        return self.final[name]

    @contextmanager
    def open(self, name: str, mode: str = "wb") -> Iterator[Any]:
        """Open output ``name``'s temp file, fsyncing it on close."""
        target = self.path(name)
        handle = open(target, mode)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

    def staged_paths(self) -> dict[str, Path]:
        return dict(self.staged)

    def missing(self) -> list[str]:
        return [name for name, p in self.staged.items() if not p.exists()]

    def commit(self) -> dict[str, Path]:
        """fsync and rename every staged output into its final path."""
        missing = self.missing()
        if missing:
            raise StageError(
                f"stage {self.stage!r} did not write declared output(s): {missing}"
            )
        for name, staged in self.staged.items():
            os.replace(staged, self.final[name])
        for parent in {p.parent for p in self.final.values()}:
            _fsync_dir(parent)
        self._committed = True
        return dict(self.final)

    def discard(self) -> None:
        """Remove staged files; the final paths are left untouched."""
        for staged in self.staged.values():
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
            except IsADirectoryError:  # pragma: no cover - directory outputs
                pass


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
@dataclass
class StageRun:
    """Handle passed to a stage body while it executes."""

    stage: "Stage"
    options: RunOptions
    inputs: dict[str, Path]
    config: dict[str, Any]
    progress: ProgressReporter
    log: Callable[[str], None]


class Stage:
    """A restartable unit of work with fingerprinted completion.

    Subclass and override :meth:`execute` (and optionally
    :meth:`validate_outputs`), or use :class:`FunctionStage`.
    """

    def __init__(
        self,
        name: str,
        *,
        inputs: Mapping[str, str | os.PathLike[str]] | None = None,
        outputs: Mapping[str, str | os.PathLike[str]] | None = None,
        config: Mapping[str, Any] | None = None,
        marker_dir: str | os.PathLike[str] = ".stages",
        log_path: str | os.PathLike[str] | None = None,
        progress_sink: Callable[[str], None] | None = None,
        progress_total: int | None = None,
        progress_unit: str = "items",
        progress_interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not name or any(ch in name for ch in "/\\"):
            raise ValueError(f"stage name must be a non-empty path-safe label, got {name!r}")
        self.name = name
        self.inputs: dict[str, Path] = {k: Path(v) for k, v in (inputs or {}).items()}
        self.outputs: dict[str, Path] = {k: Path(v) for k, v in (outputs or {}).items()}
        self.config: dict[str, Any] = dict(config or {})
        self.marker_dir = Path(marker_dir)
        self.log_path = Path(log_path) if log_path is not None else None
        self.progress_sink = progress_sink
        self.progress_total = progress_total
        self.progress_unit = progress_unit
        self.progress_interval_s = progress_interval_s
        self.clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))

    # -- paths ----------------------------------------------------------- #
    @property
    def marker_path(self) -> Path:
        return self.marker_dir / f"{self.name}{MARKER_SUFFIX}"

    # -- logging --------------------------------------------------------- #
    def log(self, message: str) -> None:
        """Append a redacted, timestamped line to the stage log (and sink).

        Everything goes through :func:`crater.download.redact_text`, so a
        presigned archive URL that reaches a stage log is stripped of its
        signature before it is written to a file that may be committed.
        """
        line = f"{self._now().isoformat(timespec='seconds')} [{self.name}] {redact_text(message)}"
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.progress_sink is not None:
            self.progress_sink(line)

    # -- fingerprinting -------------------------------------------------- #
    def missing_inputs(self) -> list[str]:
        return [name for name, path in sorted(self.inputs.items()) if not path.exists()]

    def fingerprint(self) -> Fingerprint:
        """Current fingerprint.  Raises ``FileNotFoundError`` if inputs are missing."""
        return Fingerprint(
            stage=self.name,
            inputs=fingerprint_inputs(self.inputs),
            config_hash=fingerprint_config(self.config),
        )

    # -- marker ---------------------------------------------------------- #
    def read_marker(self) -> dict[str, Any] | None:
        """Parsed marker, or ``None`` when absent or unreadable."""
        path = self.marker_path
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "fingerprint" not in data:
            return None
        return data

    def marker_state(
        self, current: Fingerprint | None = None, *, verify_output_hashes: bool = True
    ) -> tuple[str, str]:
        """``(state, reason)`` for the stage's completion marker.

        The state is :data:`MARKER_COMPLETE` only when the marker parses, its
        fingerprint matches ``current``, and every output it recorded is still
        present with the recorded size (and digest, unless
        ``verify_output_hashes`` is false).
        """
        data = self.read_marker()
        if data is None:
            if self.marker_path.exists():
                return MARKER_UNREADABLE, f"marker at {self.marker_path} is not readable JSON"
            return MARKER_MISSING, "no completion marker"

        try:
            recorded = Fingerprint.from_json(data["fingerprint"])
        except (KeyError, TypeError, ValueError):
            return MARKER_UNREADABLE, "marker fingerprint block is malformed"

        if recorded.schema_version != MARKER_SCHEMA_VERSION:
            return (
                MARKER_UNREADABLE,
                f"marker schema {recorded.schema_version} != {MARKER_SCHEMA_VERSION}",
            )

        if current is not None and recorded.digest != current.digest:
            diffs = current.differences(recorded)
            if any(d.startswith("config hash") for d in diffs):
                state = MARKER_STALE_CONFIG
            elif any(d.startswith("stage name") for d in diffs):
                state = MARKER_STALE_STAGE
            else:
                state = MARKER_STALE_INPUTS
            return state, "; ".join(diffs)

        for name, record in (data.get("outputs") or {}).items():
            path = Path(record.get("path", ""))
            if not path.exists():
                return MARKER_OUTPUT_MISSING, f"recorded output {name!r} is gone: {path}"
            size = path.stat().st_size
            if int(record.get("size_bytes", -1)) != size:
                return (
                    MARKER_OUTPUT_CHANGED,
                    f"output {name!r} is {size} bytes, marker recorded "
                    f"{record.get('size_bytes')}",
                )
            if verify_output_hashes and record.get("sha256"):
                if hash_file(path) != record["sha256"]:
                    return MARKER_OUTPUT_CHANGED, f"output {name!r} content differs from marker"
        return MARKER_COMPLETE, f"marker matches fingerprint {recorded.digest[:12]}"

    def is_complete(self, *, verify_output_hashes: bool = True) -> bool:
        """True when this stage can be skipped on resume."""
        if self.missing_inputs():
            return False
        state, _ = self.marker_state(
            self.fingerprint(), verify_output_hashes=verify_output_hashes
        )
        return state == MARKER_COMPLETE

    def invalidate(self, reason: str = "") -> bool:
        """Delete the completion marker.  Returns True if one was removed."""
        try:
            self.marker_path.unlink()
        except FileNotFoundError:
            return False
        self.log(f"completion marker invalidated{': ' + reason if reason else ''}")
        return True

    def _write_marker(self, fingerprint: Fingerprint, duration_s: float,
                      progress: Progress | None) -> None:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "stage": self.name,
            "completed_at": self._now().isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 6),
            "fingerprint": fingerprint.to_json(),
            "outputs": {
                name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": hash_file(path),
                }
                for name, path in sorted(self.outputs.items())
            },
            "log_path": str(self.log_path) if self.log_path else None,
            "progress": None if progress is None else {
                "done": progress.done, "total": progress.total, "unit": progress.unit,
            },
        }
        tmp = self.marker_path.with_name(self.marker_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.marker_path)
        _fsync_dir(self.marker_path.parent)

    # -- the work -------------------------------------------------------- #
    def execute(self, writer: OutputWriter, run: StageRun) -> None:
        """Do the work, writing to ``writer.path(name)`` temp paths."""
        raise NotImplementedError(f"stage {self.name!r} has no execute()")

    def validate_outputs(self, paths: Mapping[str, Path]) -> tuple[bool, str]:
        """Validate outputs at ``paths`` (temp paths pre-commit, final paths otherwise).

        The default demands that every declared output exists and is non-empty.
        Override for real content checks -- e.g. open a GeoTIFF with rasterio,
        or run :func:`crater.manifest.validate_manifest` over a manifest output.
        """
        problems = []
        for name, path in sorted(paths.items()):
            if not Path(path).exists():
                problems.append(f"{name} missing")
            elif Path(path).stat().st_size == 0:
                problems.append(f"{name} is empty")
        if problems:
            return False, "; ".join(problems)
        return True, f"all {len(paths)} declared output(s) present and non-empty"

    # -- driver ---------------------------------------------------------- #
    def run(self, options: RunOptions | None = None) -> StageResult:
        """Run (or skip, or inspect) the stage.

        Returns a :class:`StageResult`.  A *planning* problem -- missing inputs,
        failed output validation -- is reported as :data:`ACTION_FAILED` with a
        reason.  An exception raised by the stage body is **not** swallowed: it
        propagates after staged outputs are discarded and with no marker
        written, so a bug never looks like a completed stage.
        """
        options = options or RunOptions()

        missing = self.missing_inputs()
        if missing:
            reason = (
                "cannot run: declared input(s) missing: "
                + ", ".join(f"{n}={self.inputs[n]}" for n in missing)
            )
            self.log(reason)
            return StageResult(
                stage=self.name, action=ACTION_FAILED, ok=False, reason=reason,
                marker_state=MARKER_MISSING, fingerprint="", log_path=self.log_path,
            )

        fingerprint = self.fingerprint()
        state, state_reason = self.marker_state(
            fingerprint, verify_output_hashes=options.verify_output_hashes
        )

        # --- validate-only: read-only report -------------------------------- #
        if options.validate_only:
            valid, valid_reason = self.validate_outputs(dict(self.outputs))
            if state == MARKER_COMPLETE and valid:
                reason = f"outputs validate and {state_reason}"
                action, ok = ACTION_VALIDATED, True
            else:
                bits = [] if valid else [f"output validation failed: {valid_reason}"]
                if state != MARKER_COMPLETE:
                    bits.append(f"{state}: {state_reason}")
                reason = "; ".join(bits)
                action, ok = ACTION_INVALID, False
            self.log(f"validate-only -> {action}: {reason}")
            return StageResult(
                stage=self.name, action=action, ok=ok, reason=reason, marker_state=state,
                fingerprint=fingerprint.digest, outputs=dict(self.outputs),
                log_path=self.log_path,
            )

        # --- resume: skip only a genuinely complete stage -------------------- #
        if state == MARKER_COMPLETE and options.resume and not options.force:
            prefix = "dry run: would skip, " if options.dry_run else ""
            reason = f"{prefix}already complete; {state_reason}"
            self.log(f"skipping: {reason}")
            return StageResult(
                stage=self.name, action=ACTION_SKIPPED, ok=True, reason=reason,
                marker_state=state, fingerprint=fingerprint.digest,
                outputs=dict(self.outputs), log_path=self.log_path,
            )

        invalidation_reasons: tuple[str, ...] = ()
        if state in STALE_MARKER_STATES:
            invalidation_reasons = tuple(part.strip() for part in state_reason.split(";"))

        # --- dry run: report, touch nothing --------------------------------- #
        if options.dry_run:
            if state == MARKER_COMPLETE:
                reason = f"would re-execute (force requested); {state_reason}"
            elif state == MARKER_MISSING:
                reason = "would execute: no completion marker"
            else:
                reason = (
                    f"would invalidate the completion marker and re-execute: "
                    f"{state} ({state_reason})"
                )
            self.log(f"dry run -> {reason}")
            return StageResult(
                stage=self.name, action=ACTION_WOULD_EXECUTE, ok=True, reason=reason,
                marker_state=state, fingerprint=fingerprint.digest,
                outputs=dict(self.outputs), invalidated=False,
                invalidation_reasons=invalidation_reasons, log_path=self.log_path,
            )

        # --- real execution -------------------------------------------------- #
        invalidated = False
        if state in STALE_MARKER_STATES or (state == MARKER_COMPLETE and options.force):
            invalidated = self.invalidate(
                state_reason if state != MARKER_COMPLETE else "force requested"
            )
            if state == MARKER_COMPLETE and options.force:
                invalidation_reasons = ("force requested",)

        reporter = ProgressReporter(
            total=self.progress_total, unit=self.progress_unit, label=self.name,
            sink=self.log, clock=self.clock, min_interval_s=self.progress_interval_s,
        )
        writer = OutputWriter(self.name, self.outputs)
        run_ctx = StageRun(
            stage=self, options=options, inputs=dict(self.inputs),
            config=dict(self.config), progress=reporter, log=self.log,
        )
        started = self.clock()
        reporter.start()
        try:
            self.execute(writer, run_ctx)
        except BaseException:
            writer.discard()
            self.log("stage body raised; staged outputs discarded, no marker written")
            raise
        reporter.finish()
        duration = self.clock() - started

        missing_outputs = writer.missing()
        if missing_outputs:
            writer.discard()
            reason = f"stage did not write declared output(s): {missing_outputs}"
            self.log(f"failed: {reason}")
            return StageResult(
                stage=self.name, action=ACTION_FAILED, ok=False, reason=reason,
                marker_state=MARKER_MISSING, fingerprint=fingerprint.digest,
                invalidated=invalidated, invalidation_reasons=invalidation_reasons,
                duration_s=duration, progress=reporter.snapshot(), log_path=self.log_path,
            )

        valid, valid_reason = self.validate_outputs(writer.staged_paths())
        if not valid:
            writer.discard()
            reason = f"output validation failed before commit: {valid_reason}"
            self.log(f"failed: {reason}")
            return StageResult(
                stage=self.name, action=ACTION_FAILED, ok=False, reason=reason,
                marker_state=MARKER_MISSING, fingerprint=fingerprint.digest,
                invalidated=invalidated, invalidation_reasons=invalidation_reasons,
                duration_s=duration, progress=reporter.snapshot(), log_path=self.log_path,
            )

        committed = writer.commit()
        self._write_marker(fingerprint, duration, reporter.snapshot())
        reason = f"executed and validated: {valid_reason}"
        self.log(f"completed in {duration:.3f}s: {valid_reason}")
        return StageResult(
            stage=self.name, action=ACTION_EXECUTED, ok=True, reason=reason,
            marker_state=MARKER_COMPLETE, fingerprint=fingerprint.digest,
            outputs=committed, invalidated=invalidated,
            invalidation_reasons=invalidation_reasons, duration_s=duration,
            progress=reporter.snapshot(), log_path=self.log_path,
        )


class FunctionStage(Stage):
    """A :class:`Stage` whose body and validator are plain callables.

    ``run_fn(writer, run)`` writes the staged outputs; ``validate_fn(paths)``
    returns ``(ok, reason)`` and defaults to the base class's existence check.
    """

    def __init__(self, name: str, run_fn: Callable[[OutputWriter, StageRun], None],
                 *, validate_fn: Callable[[Mapping[str, Path]], tuple[bool, str]] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._run_fn = run_fn
        self._validate_fn = validate_fn

    def execute(self, writer: OutputWriter, run: StageRun) -> None:
        self._run_fn(writer, run)

    def validate_outputs(self, paths: Mapping[str, Path]) -> tuple[bool, str]:
        if self._validate_fn is None:
            return super().validate_outputs(paths)
        return self._validate_fn(paths)


# --------------------------------------------------------------------------- #
# Pipeline helper
# --------------------------------------------------------------------------- #
def run_pipeline(
    stages: list[Stage], options: RunOptions | None = None, *, stop_on_failure: bool = True
) -> list[StageResult]:
    """Run stages in order, stopping at the first failure by default.

    Downstream stages must not run on a failed upstream: their outputs would be
    derived from inputs that do not exist or did not validate.
    """
    options = options or RunOptions()
    results: list[StageResult] = []
    for stage in stages:
        result = stage.run(options)
        results.append(result)
        if not result.ok and stop_on_failure:
            break
    return results


__all__ = [
    "ACTION_EXECUTED", "ACTION_FAILED", "ACTION_INVALID", "ACTION_SKIPPED",
    "ACTION_VALIDATED", "ACTION_WOULD_EXECUTE", "ETA_NO_THROUGHPUT", "ETA_NO_TOTAL",
    "Fingerprint", "FunctionStage", "MARKER_COMPLETE", "MARKER_MISSING",
    "MARKER_OUTPUT_CHANGED", "MARKER_OUTPUT_MISSING", "MARKER_SCHEMA_VERSION",
    "MARKER_STALE_CONFIG", "MARKER_STALE_INPUTS", "MARKER_STALE_STAGE",
    "MARKER_SUFFIX", "MARKER_UNREADABLE", "OutputWriter", "Progress",
    "ProgressReporter", "RunOptions", "STALE_MARKER_STATES", "Stage",
    "StageConfigError", "StageError", "StageResult", "StageRun", "canonical_json",
    "fingerprint_config", "fingerprint_inputs", "hash_file", "run_pipeline",
]
