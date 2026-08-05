"""Contains tests for the ProcessingTracker, ProcessingStatus, and JobState classes."""

import os
from pathlib import Path

import yaml
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker


def test_processing_tracker_initialization(tmp_path: Path) -> None:
    """Verifies basic initialization of ProcessingTracker."""
    tracker_file = tmp_path / "test_tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    assert tracker.file_path == tracker_file
    assert tracker.jobs == {}
    assert tracker.lock_path == str(tracker_file.with_suffix(".yaml.lock"))


def test_processing_tracker_generate_job_id() -> None:
    """Verifies that generate_job_id produces consistent hash-based IDs."""
    job_name = "suite2p_processing"
    specifier = "plane_0"

    first_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
    second_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)

    assert first_id == second_id
    assert len(first_id) == 16
    assert all(character in "0123456789abcdef" for character in first_id)


def test_processing_tracker_generate_job_id_unique() -> None:
    """Verifies that different jobs produce different IDs."""
    # Different job names produce different IDs.
    first_id = ProcessingTracker.generate_job_id(job_name="job1", specifier="")
    second_id = ProcessingTracker.generate_job_id(job_name="job2", specifier="")
    assert first_id != second_id

    # The same job name with different specifiers produces different IDs.
    third_id = ProcessingTracker.generate_job_id(job_name="process_plane", specifier="plane_0")
    fourth_id = ProcessingTracker.generate_job_id(job_name="process_plane", specifier="plane_1")
    assert third_id != fourth_id


def test_processing_tracker_generate_job_id_without_specifier() -> None:
    """Verifies that generate_job_id works without a specifier."""
    job_name = "suite2p_processing"

    # With empty specifier.
    first_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier="")
    # Without specifier (default).
    second_id = ProcessingTracker.generate_job_id(job_name=job_name)

    assert first_id == second_id
    assert len(first_id) == 16


def test_processing_tracker_initialize_jobs(tmp_path: Path) -> None:
    """Verifies that initialize_jobs creates scheduled job entries and returns job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [
        ("job1", ""),
        ("job2", ""),
        ("job3", ""),
    ]

    job_ids = tracker.initialize_jobs(jobs=jobs)

    # Verifies returned job IDs.
    assert len(job_ids) == 3
    for (job_name, specifier), job_id in zip(jobs, job_ids, strict=True):
        assert job_id == ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)

    # Reloads to verify persistence.
    tracker._load_state()
    assert len(tracker.jobs) == 3
    for job_name, specifier in jobs:
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        assert job_id in tracker.jobs
        assert tracker.jobs[job_id].job_name == job_name
        assert tracker.jobs[job_id].specifier == specifier
        assert tracker.jobs[job_id].status == ProcessingStatus.SCHEDULED
        assert tracker.jobs[job_id].executor_id is None


def test_processing_tracker_initialize_jobs_preserves_existing(tmp_path: Path) -> None:
    """Verifies that initialize_jobs preserves existing job entries."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [
        ("job1", ""),
        ("job2", ""),
    ]
    job_ids = [ProcessingTracker.generate_job_id(job_name=name, specifier=specifier) for name, specifier in jobs]

    # Initializes first time.
    tracker.initialize_jobs(jobs=jobs)

    # Gives the first job a non-default status before reinitializing.
    tracker.start_job(job_id=job_ids[0])

    # Reinitializes with the same jobs.
    tracker.initialize_jobs(jobs=jobs)

    # Verifies the first job's status is preserved.
    tracker._load_state()
    assert tracker.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED


def test_processing_tracker_find_jobs(tmp_path: Path) -> None:
    """Verifies that find_jobs searches by name and specifier with partial matching."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [
        ("process_plane", "plane_0"),
        ("process_plane", "plane_1"),
        ("process_plane", "plane_2"),
        ("extract_signals", ""),
        ("suite2p_registration", "batch_1"),
    ]
    tracker.initialize_jobs(jobs=jobs)

    # Searches by job name only.
    matches = tracker.find_jobs(job_name="process_plane")
    assert len(matches) == 3

    # Searches by specifier only.
    matches = tracker.find_jobs(specifier="plane_1")
    assert len(matches) == 1
    assert next(iter(matches.values())) == ("process_plane", "plane_1")

    # Searches by partial job name.
    matches = tracker.find_jobs(job_name="process")
    assert len(matches) == 3

    # Searches by partial specifier.
    matches = tracker.find_jobs(specifier="plane")
    assert len(matches) == 3

    # Searches by both name and specifier.
    matches = tracker.find_jobs(job_name="process_plane", specifier="plane_0")
    assert len(matches) == 1

    # No matches.
    matches = tracker.find_jobs(job_name="nonexistent")
    assert not matches


def test_processing_tracker_find_jobs_without_arguments(tmp_path: Path) -> None:
    """Verifies that find_jobs called without arguments matches every tracked job."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("process_plane", "plane_0"), ("process_plane", "plane_1"), ("extract_signals", "")]
    tracker.initialize_jobs(jobs=jobs)

    assert len(tracker.find_jobs()) == len(jobs)


def test_processing_tracker_snapshot(tmp_path: Path) -> None:
    """Verifies that snapshot returns the full job registry as detached copies."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job_a", ""), ("job_b", "1")])

    job_id = ProcessingTracker.generate_job_id(job_name="job_a", specifier="")
    tracker.start_job(job_id=job_id, executor_id="executor-1")
    tracker.complete_job(job_id=job_id)

    snapshot = tracker.snapshot()
    assert len(snapshot) == 2
    assert snapshot[job_id].status == ProcessingStatus.SUCCEEDED
    assert snapshot[job_id].executor_id == "executor-1"
    assert snapshot[job_id].started_at is not None
    assert snapshot[job_id].completed_at is not None

    # Confirms that mutating a snapshot does not affect the tracker or leak into a later save.
    snapshot[job_id].status = ProcessingStatus.FAILED
    assert tracker.snapshot()[job_id].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_reset_jobs_preserves_untargeted_state(tmp_path: Path) -> None:
    """Verifies that reset_jobs clears only the targeted jobs and leaves sibling history byte-identical."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    jobs = [("phase_a", ""), ("phase_b", "1"), ("phase_b", "2")]
    tracker.initialize_jobs(jobs=jobs)

    for job_name, specifier in jobs:
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        tracker.start_job(job_id=job_id, executor_id="executor-7")
        tracker.complete_job(job_id=job_id)

    before = tracker.snapshot()
    target = ProcessingTracker.generate_job_id(job_name="phase_b", specifier="1")

    assert tracker.reset_jobs(job_ids=[target]) == [target]

    after = tracker.snapshot()
    assert after[target].status == ProcessingStatus.SCHEDULED
    assert after[target].executor_id is None
    assert after[target].started_at is None
    assert after[target].completed_at is None

    # Confirms that every other job keeps its recorded outcome, executor, and timing.
    for job_id in (identifier for identifier in before if identifier != target):
        assert after[job_id] == before[job_id]


def test_processing_tracker_reset_jobs_rejects_unknown_ids(tmp_path: Path) -> None:
    """Verifies that reset_jobs rejects unknown job IDs and leaves the tracker untouched."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job_a", "")])

    job_id = ProcessingTracker.generate_job_id(job_name="job_a", specifier="")
    tracker.start_job(job_id=job_id)
    tracker.complete_job(job_id=job_id)

    with pytest.raises(ValueError, match="0000000000000000"):
        tracker.reset_jobs(job_ids=[job_id, "0000000000000000"])

    # Confirms that the recognized job was not reset by the rejected request.
    assert tracker.snapshot()[job_id].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_snapshot_does_not_create_missing_tracker(tmp_path: Path) -> None:
    """Verifies that snapshotting an absent tracker yields an empty registry without creating the file."""
    tracker_file = tmp_path / "absent_tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    assert tracker.snapshot() == {}
    assert not tracker_file.exists()


def test_processing_tracker_align_jobs(tmp_path: Path) -> None:
    """Verifies that align_jobs initializes, additively registers, and stays idempotent."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    universe = [("job_a", ""), ("job_b", "1"), ("job_b", "2")]

    # Initializes from scratch against a subset, then records an outcome on it.
    assert len(tracker.align_jobs(jobs=[("job_a", "")], universe=universe)) == 1
    job_id = ProcessingTracker.generate_job_id(job_name="job_a", specifier="")
    tracker.start_job(job_id=job_id)
    tracker.complete_job(job_id=job_id)

    # Additively registers the missing universe members without disturbing the recorded outcome.
    assert len(tracker.align_jobs(jobs=universe, universe=universe)) == len(universe)
    snapshot = tracker.snapshot()
    assert len(snapshot) == len(universe)
    assert snapshot[job_id].status == ProcessingStatus.SUCCEEDED

    # Confirms that re-aligning an already-aligned registry is a no-op.
    tracker.align_jobs(jobs=universe, universe=universe)
    assert tracker.snapshot()[job_id].status == ProcessingStatus.SUCCEEDED
    assert len(tracker.snapshot()) == len(universe)


def test_processing_tracker_align_jobs_discards_foreign_entries(tmp_path: Path) -> None:
    """Verifies that align_jobs rebuilds the registry when it holds jobs outside the requested universe."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    tracker.align_jobs(jobs=[("legacy_job", "x")])
    legacy_id = ProcessingTracker.generate_job_id(job_name="legacy_job", specifier="x")
    assert legacy_id in tracker.snapshot()

    universe = [("job_a", ""), ("job_b", "1")]
    tracker.align_jobs(jobs=universe, universe=universe)

    snapshot = tracker.snapshot()
    assert legacy_id not in snapshot
    assert len(snapshot) == len(universe)


def test_processing_tracker_align_jobs_defaults_universe_to_jobs(tmp_path: Path) -> None:
    """Verifies that omitting the universe treats the requested jobs as the full universe."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    jobs = [("job_a", ""), ("job_b", "1")]

    assert len(tracker.align_jobs(jobs=jobs)) == len(jobs)
    assert len(tracker.snapshot()) == len(jobs)


def test_processing_tracker_start_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that start_job marks a job as running and falls back to the scheme-tagged process ID."""
    _clear_scheduler_environment(monkeypatch=monkeypatch)
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_id].executor_id == f"pid:{os.getpid()}"


def test_processing_tracker_start_job_resolves_slurm_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that start_job auto-resolves the executor ID to the scheme-tagged SLURM job ID under SLURM."""
    _clear_scheduler_environment(monkeypatch=monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].executor_id == "slurm:987654"


def test_processing_tracker_resolve_executor_id_reads_slurm_jobid_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the resolver honors the older SLURM_JOBID variable when SLURM_JOB_ID is absent."""
    _clear_scheduler_environment(monkeypatch=monkeypatch)
    monkeypatch.setenv("SLURM_JOBID", "555")

    assert ProcessingTracker._resolve_executor_id() == "slurm:555"


def test_processing_tracker_resolve_executor_id_tags_non_slurm_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the resolver tags a non-SLURM scheduler's job id with the matching scheme."""
    _clear_scheduler_environment(monkeypatch=monkeypatch)
    monkeypatch.setenv("PBS_JOBID", "42.headnode")

    assert ProcessingTracker._resolve_executor_id() == "pbs:42.headnode"


def test_processing_tracker_resolve_executor_id_requires_sge_corroboration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the generic JOB_ID variable is accepted as Grid Engine only when SGE_ROOT corroborates it."""
    _clear_scheduler_environment(monkeypatch=monkeypatch)

    # Confirms that a bare JOB_ID from an unrelated tool is not mistaken for a Grid Engine allocation.
    monkeypatch.setenv("JOB_ID", "not-a-scheduler")
    assert ProcessingTracker._resolve_executor_id() == f"pid:{os.getpid()}"

    # Confirms that the same JOB_ID is recognized as a Grid Engine job once SGE_ROOT is present.
    monkeypatch.setenv("SGE_ROOT", "/opt/sge")
    assert ProcessingTracker._resolve_executor_id() == "sge:not-a-scheduler"


def test_processing_tracker_start_job_with_executor_id(tmp_path: Path) -> None:
    """Verifies that start_job records an explicitly provided executor_id without resolving from the environment."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id, executor_id="slurm-12345")

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_id].executor_id == "slurm-12345"


def test_processing_tracker_start_job_raises_for_unknown_job(tmp_path: Path) -> None:
    """Verifies that start_job raises ValueError for unknown job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    unknown_job_id = "nonexistent_job_id"

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.start_job(job_id=unknown_job_id)


def test_processing_tracker_complete_job(tmp_path: Path) -> None:
    """Verifies that complete_job marks a job as succeeded."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)
    tracker.complete_job(job_id=job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_fail_job(tmp_path: Path) -> None:
    """Verifies that fail_job marks a job as failed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)
    tracker.fail_job(job_id=job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.FAILED


def test_processing_tracker_get_job_status(tmp_path: Path) -> None:
    """Verifies that get_job_status returns the correct status."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    tracker.initialize_jobs(jobs=[("test_job", "")])

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SCHEDULED

    # Starts and checks the running status.
    tracker.start_job(job_id=job_id)
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.RUNNING

    # Completes and checks the succeeded status.
    tracker.complete_job(job_id=job_id)
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


def test_processing_tracker_reset(tmp_path: Path) -> None:
    """Verifies that reset clears all jobs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(job_name=name, specifier=specifier) for name, specifier in jobs]

    tracker.initialize_jobs(jobs=jobs)
    tracker.start_job(job_id=job_ids[0])

    tracker.reset()

    tracker._load_state()
    assert not tracker.jobs


def test_processing_tracker_complete_property(tmp_path: Path) -> None:
    """Verifies that the complete property returns True when all jobs succeed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(job_name=name, specifier=specifier) for name, specifier in jobs]

    tracker.initialize_jobs(jobs=jobs)
    assert not tracker.complete

    tracker.start_job(job_id=job_ids[0])
    tracker.complete_job(job_id=job_ids[0])
    assert not tracker.complete

    tracker.start_job(job_id=job_ids[1])
    tracker.complete_job(job_id=job_ids[1])
    assert tracker.complete


def test_processing_tracker_encountered_error_property(tmp_path: Path) -> None:
    """Verifies that the encountered_error property returns True when any job fails."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(job_name=name, specifier=specifier) for name, specifier in jobs]

    tracker.initialize_jobs(jobs=jobs)
    assert not tracker.encountered_error

    tracker.start_job(job_id=job_ids[0])
    tracker.complete_job(job_id=job_ids[0])
    assert not tracker.encountered_error

    tracker.start_job(job_id=job_ids[1])
    tracker.fail_job(job_id=job_ids[1])
    assert tracker.encountered_error


def test_processing_tracker_concurrent_access(tmp_path: Path) -> None:
    """Verifies that file locks prevent race conditions."""
    tracker_file = tmp_path / "tracker.yaml"

    # Simulates two processes.
    first_tracker = ProcessingTracker(file_path=tracker_file)
    second_tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")

    # Initializes from the first process.
    first_tracker.initialize_jobs(jobs=[("test_job", "")])

    # Confirms the second process sees the job.
    assert second_tracker.get_job_status(job_id=job_id) == ProcessingStatus.SCHEDULED

    # Starts the job from the first process.
    first_tracker.start_job(job_id=job_id)

    # Confirms the second process sees the update.
    assert second_tracker.get_job_status(job_id=job_id) == ProcessingStatus.RUNNING


def test_processing_tracker_yaml_serialization(tmp_path: Path) -> None:
    """Verifies that the tracker state is properly serialized to YAML."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "specifier")]
    job_ids = [ProcessingTracker.generate_job_id(job_name=name, specifier=specifier) for name, specifier in jobs]

    tracker.initialize_jobs(jobs=jobs)
    tracker.start_job(job_id=job_ids[0])

    # Creates a new instance and verifies it loads correctly.
    reloaded_tracker = ProcessingTracker(file_path=tracker_file)
    reloaded_tracker._load_state()

    assert len(reloaded_tracker.jobs) == 2
    assert reloaded_tracker.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert reloaded_tracker.jobs[job_ids[0]].job_name == "job1"
    assert reloaded_tracker.jobs[job_ids[0]].specifier == ""
    assert reloaded_tracker.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED
    assert reloaded_tracker.jobs[job_ids[1]].job_name == "job2"
    assert reloaded_tracker.jobs[job_ids[1]].specifier == "specifier"


def test_processing_status_enum_values() -> None:
    """Verifies all ProcessingStatus enumeration values."""
    assert ProcessingStatus.SCHEDULED == 0
    assert ProcessingStatus.RUNNING == 1
    assert ProcessingStatus.SUCCEEDED == 2
    assert ProcessingStatus.FAILED == 3


def test_job_state_defaults() -> None:
    """Verifies default JobState initialization."""
    job = JobState(job_name="test_job")
    assert job.job_name == "test_job"
    assert job.specifier == ""
    assert job.status == ProcessingStatus.SCHEDULED
    assert job.executor_id is None


def test_job_state_with_specifier() -> None:
    """Verifies JobState initialization with specifier."""
    job = JobState(job_name="process_plane", specifier="plane_0")
    assert job.job_name == "process_plane"
    assert job.specifier == "plane_0"
    assert job.status == ProcessingStatus.SCHEDULED


def test_job_state_with_executor_id() -> None:
    """Verifies JobState initialization with executor_id."""
    job = JobState(job_name="test_job", status=ProcessingStatus.RUNNING, executor_id="pid-42")
    assert job.job_name == "test_job"
    assert job.status == ProcessingStatus.RUNNING
    assert job.executor_id == "pid-42"


def test_job_state_new_fields() -> None:
    """Verifies JobState initialization with error_message and timestamp fields."""
    job = JobState(
        job_name="test_job",
        status=ProcessingStatus.FAILED,
        error_message="Out of memory",
        started_at=1234567890123456,
        completed_at=1234567890234567,
    )
    assert job.error_message == "Out of memory"
    assert job.started_at == 1234567890123456
    assert job.completed_at == 1234567890234567


def test_processing_tracker_timestamps(tmp_path: Path) -> None:
    """Verifies that start_job and complete_job set timestamps."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")
    tracker.initialize_jobs(jobs=[("test_job", "")])

    # Starts the job and verifies started_at is set.
    tracker.start_job(job_id=job_id)
    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.started_at is not None
    assert job_info.completed_at is None

    # Completes the job and verifies completed_at is set.
    tracker.complete_job(job_id=job_id)
    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.completed_at is not None
    assert job_info.completed_at >= job_info.started_at


def test_processing_tracker_fail_job_with_error_message(tmp_path: Path) -> None:
    """Verifies that fail_job records the error message."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")
    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)
    tracker.fail_job(job_id=job_id, error_message="CUDA out of memory")

    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.status == ProcessingStatus.FAILED
    assert job_info.error_message == "CUDA out of memory"
    assert job_info.completed_at is not None


def test_processing_tracker_start_job_clears_previous_attempt(tmp_path: Path) -> None:
    """Verifies that start_job clears the error message and completion timestamp left by a previous attempt."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")
    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)
    tracker.fail_job(job_id=job_id, error_message="CUDA out of memory")

    tracker.start_job(job_id=job_id)

    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.status == ProcessingStatus.RUNNING
    assert job_info.error_message is None
    assert job_info.completed_at is None


def test_processing_tracker_complete_job_clears_previous_error(tmp_path: Path) -> None:
    """Verifies that complete_job clears the error message when a previously failed job succeeds."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")
    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id=job_id)
    tracker.fail_job(job_id=job_id, error_message="CUDA out of memory")

    tracker.complete_job(job_id=job_id)

    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.status == ProcessingStatus.SUCCEEDED
    assert job_info.error_message is None


def test_processing_tracker_align_jobs_preserves_in_universe_siblings(tmp_path: Path) -> None:
    """Verifies that discarding foreign entries leaves the recorded state of every in-universe job intact."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    universe = [("extract", "101"), ("extract", "152"), ("parse", "101-3-1")]
    job_ids = tracker.align_jobs(jobs=universe, universe=universe)
    for job_id in job_ids:
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    # Simulates an entry left over from an earlier pipeline definition.
    tracker.initialize_jobs(jobs=[("legacy_job", "")])
    tracker.align_jobs(jobs=[("parse", "101-3-1")], universe=universe)

    registry = tracker.snapshot()
    assert set(registry) == set(job_ids)
    assert registry[job_ids[0]].status == ProcessingStatus.SUCCEEDED
    assert registry[job_ids[1]].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_align_jobs_rejects_out_of_universe_request(tmp_path: Path) -> None:
    """Verifies that requesting a job outside the declared universe raises and leaves the tracker untouched."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    universe = [("extract", "101"), ("parse", "101-3-1")]
    job_ids = tracker.align_jobs(jobs=universe, universe=universe)
    for job_id in job_ids:
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    error_message = (
        f"Unable to align the processing tracker at '{tracker_file}' with the requested jobs. Every requested job "
        f"must be part of the declared job universe, but the following are absent from it: typo_job (999)."
    )

    with pytest.raises(ValueError, match=error_format(error_message)):
        tracker.align_jobs(jobs=[("typo_job", "999")], universe=universe)

    registry = tracker.snapshot()
    assert set(registry) == set(job_ids)
    assert all(job_state.status == ProcessingStatus.SUCCEEDED for job_state in registry.values())


def test_processing_tracker_get_job_info_returns_a_copy(tmp_path: Path) -> None:
    """Verifies that mutating the state returned by get_job_info leaves the tracker's registry unchanged."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="test_job", specifier="")
    tracker.initialize_jobs(jobs=[("test_job", "")])

    job_info = tracker.get_job_info(job_id=job_id)
    job_info.status = ProcessingStatus.SUCCEEDED
    job_info.error_message = "mutated by the caller"

    assert tracker.get_job_info(job_id=job_id).status == ProcessingStatus.SCHEDULED
    assert tracker.get_job_info(job_id=job_id).error_message is None


def test_processing_tracker_get_jobs_by_status(tmp_path: Path) -> None:
    """Verifies that get_jobs_by_status returns correct job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", ""), ("job4", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    scheduled = tracker.get_jobs_by_status(status=ProcessingStatus.SCHEDULED)
    assert len(scheduled) == 4

    tracker.start_job(job_id=job_ids[0])
    tracker.start_job(job_id=job_ids[1])
    running = tracker.get_jobs_by_status(status=ProcessingStatus.RUNNING)
    assert len(running) == 2
    assert job_ids[0] in running
    assert job_ids[1] in running

    # Completes one, fails one.
    tracker.complete_job(job_id=job_ids[0])
    tracker.fail_job(job_id=job_ids[1])
    succeeded = tracker.get_jobs_by_status(status=ProcessingStatus.SUCCEEDED)
    failed = tracker.get_jobs_by_status(status=ProcessingStatus.FAILED)
    assert len(succeeded) == 1
    assert len(failed) == 1


def test_processing_tracker_get_summary(tmp_path: Path) -> None:
    """Verifies that get_summary returns correct counts."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", ""), ("job4", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    summary = tracker.get_summary()
    assert summary[ProcessingStatus.SCHEDULED] == 4
    assert summary[ProcessingStatus.RUNNING] == 0
    assert summary[ProcessingStatus.SUCCEEDED] == 0
    assert summary[ProcessingStatus.FAILED] == 0

    # Mixed states.
    tracker.start_job(job_id=job_ids[0])
    tracker.start_job(job_id=job_ids[1])
    tracker.complete_job(job_id=job_ids[0])
    tracker.fail_job(job_id=job_ids[1])

    summary = tracker.get_summary()
    assert summary[ProcessingStatus.SCHEDULED] == 2
    assert summary[ProcessingStatus.RUNNING] == 0
    assert summary[ProcessingStatus.SUCCEEDED] == 1
    assert summary[ProcessingStatus.FAILED] == 1


def test_processing_tracker_get_job_info(tmp_path: Path) -> None:
    """Verifies that get_job_info returns the full JobState."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id(job_name="process_plane", specifier="plane_0")
    tracker.initialize_jobs(jobs=[("process_plane", "plane_0")])
    tracker.start_job(job_id=job_id, executor_id="slurm-12345")

    job_info = tracker.get_job_info(job_id=job_id)
    assert job_info.job_name == "process_plane"
    assert job_info.specifier == "plane_0"
    assert job_info.status == ProcessingStatus.RUNNING
    assert job_info.executor_id == "slurm-12345"
    assert job_info.started_at is not None


def test_processing_tracker_get_job_info_raises_for_unknown(tmp_path: Path) -> None:
    """Verifies that get_job_info raises ValueError for unknown job."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("test_job", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.get_job_info(job_id="nonexistent_id")


def test_processing_tracker_retry_failed_jobs(tmp_path: Path) -> None:
    """Verifies that retry_failed_jobs resets failed jobs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    tracker.start_job(job_id=job_ids[0])
    tracker.start_job(job_id=job_ids[1])
    tracker.fail_job(job_id=job_ids[0], error_message="Error 1")
    tracker.fail_job(job_id=job_ids[1], error_message="Error 2")
    tracker.start_job(job_id=job_ids[2])
    tracker.complete_job(job_id=job_ids[2])

    # Retries failed jobs.
    retried = tracker.retry_failed_jobs()
    assert len(retried) == 2
    assert job_ids[0] in retried
    assert job_ids[1] in retried

    # Verifies reset state.
    job_info = tracker.get_job_info(job_id=job_ids[0])
    assert job_info.status == ProcessingStatus.SCHEDULED
    assert job_info.error_message is None
    assert job_info.started_at is None
    assert job_info.completed_at is None
    assert job_info.executor_id is None

    # Confirms the succeeded job is not affected.
    job_info = tracker.get_job_info(job_id=job_ids[2])
    assert job_info.status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_complete_job_invalid_id(tmp_path: Path) -> None:
    """Verifies that complete_job raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.complete_job(job_id="invalid_job_id")


def test_processing_tracker_fail_job_invalid_id(tmp_path: Path) -> None:
    """Verifies that fail_job raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.fail_job(job_id="invalid_job_id")


def test_processing_tracker_get_job_status_invalid_id(tmp_path: Path) -> None:
    """Verifies that get_job_status raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.get_job_status(job_id="invalid_job_id")


def test_processing_tracker_complete_property_empty_jobs(tmp_path: Path) -> None:
    """Verifies that complete property returns False for empty tracker."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    # Saves empty state.
    tracker._save_state()

    assert not tracker.complete


def test_processing_tracker_generate_job_id_golden_values() -> None:
    """Verifies that generate_job_id produces the exact digests every tracker file on disk is keyed by."""
    # These are golden values. Job IDs are the persisted keys inside every tracker YAML and the cross-process handle
    # for a job, so any change to the delimiter, the encoding, the hash function, or its seed orphans existing
    # trackers. Deriving the expectation by calling the function would make the implementation its own oracle.
    assert ProcessingTracker.generate_job_id(job_name="suite2p_processing", specifier="plane_0") == "cd0547cf71e4ea30"
    assert ProcessingTracker.generate_job_id(job_name="suite2p_processing", specifier="") == "ced6b6029ca878f9"
    assert ProcessingTracker.generate_job_id(job_name="process_data", specifier="batch_101") == "e30d3e55fffe8b95"


def test_processing_tracker_generate_job_id_rejects_a_colon() -> None:
    """Verifies that a colon in either component is rejected, since it joins them inside the hashed string."""
    message = (
        "Unable to generate the identifier for the job 'data:batch' with the specifier ''. The job name and the "
        "specifier must not contain the ':' character, as it joins them inside the hashed identifier string."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        ProcessingTracker.generate_job_id(job_name="data:batch")

    specifier_message = (
        "Unable to generate the identifier for the job 'data' with the specifier 'batch:one'. The job name and the "
        "specifier must not contain the ':' character, as it joins them inside the hashed identifier string."
    )
    with pytest.raises(ValueError, match=error_format(specifier_message)):
        ProcessingTracker.generate_job_id(job_name="data", specifier="batch:one")


def test_processing_tracker_align_jobs_rejects_an_empty_request(tmp_path: Path) -> None:
    """Verifies that an empty job list is rejected instead of resolving to a universe that discards every entry."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    universe = [("job_a", ""), ("job_b", "1")]
    job_ids = tracker.align_jobs(jobs=universe, universe=universe)
    for job_id in job_ids:
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    message = (
        f"Unable to align the processing tracker at '{tracker_file}' with the requested jobs. The 'jobs' argument "
        f"must name at least one job, but an empty list was provided."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        tracker.align_jobs(jobs=[])

    # The rejected request left every recorded outcome in place.
    registry = tracker.snapshot()
    assert set(registry) == set(job_ids)
    assert all(job_state.status == ProcessingStatus.SUCCEEDED for job_state in registry.values())


def test_processing_tracker_serializes_the_job_registry_alone(tmp_path: Path) -> None:
    """Verifies that the tracker document carries the job registry and neither of the instance's path fields."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job_a", "")])

    with tracker_file.open() as yaml_file:
        document = yaml.safe_load(yaml_file)

    assert set(document) == {"jobs"}


def test_processing_tracker_from_yaml_returns_an_attached_instance(tmp_path: Path) -> None:
    """Verifies that a tracker rebuilt through the public from_yaml() stays bound to the file it was read from."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    job_ids = tracker.initialize_jobs(jobs=[("job_a", ""), ("job_b", "")])
    tracker.start_job(job_id=job_ids[0])
    tracker.complete_job(job_id=job_ids[0])

    rebuilt = ProcessingTracker.from_yaml(file_path=tracker_file)

    assert rebuilt.file_path == tracker_file
    assert rebuilt.lock_path == str(tracker_file.with_suffix(".yaml.lock"))
    # The rebuilt instance reads the same registry the writer recorded, rather than reporting an empty pipeline.
    assert rebuilt.snapshot()[job_ids[0]].status == ProcessingStatus.SUCCEEDED
    assert len(rebuilt.find_jobs()) == 2


def _clear_scheduler_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes every scheduler variable the executor resolver consults so a test observes the process-id fallback."""
    for variable in (
        "SLURM_JOB_ID",
        "SLURM_JOBID",
        "PBS_JOBID",
        "LSB_JOBID",
        "OAR_JOB_ID",
        "JOB_ID",
        "SGE_ROOT",
        "CCP_JOBID",
        "AZ_BATCH_JOB_ID",
        "AWS_BATCH_JOB_ID",
    ):
        monkeypatch.delenv(variable, raising=False)
