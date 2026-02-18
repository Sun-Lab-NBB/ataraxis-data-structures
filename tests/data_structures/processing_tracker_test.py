"""Contains tests for the ProcessingTracker, ProcessingStatus, and JobState classes."""

import pytest

from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker


def test_processing_tracker_initialization(tmp_path):
    """Verifies basic initialization of ProcessingTracker."""
    tracker_file = tmp_path / "test_tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    assert tracker.file_path == tracker_file
    assert tracker.jobs == {}
    assert tracker.lock_path == str(tracker_file.with_suffix(".yaml.lock"))


def test_processing_tracker_generate_job_id():
    """Verifies that generate_job_id produces consistent hash-based IDs."""
    job_name = "suite2p_processing"
    specifier = "plane_0"

    # Generates the same ID multiple times
    id1 = ProcessingTracker.generate_job_id(job_name, specifier)
    id2 = ProcessingTracker.generate_job_id(job_name, specifier)

    # Should be consistent
    assert id1 == id2
    # Should be a hexadecimal string
    assert len(id1) == 16
    assert all(c in "0123456789abcdef" for c in id1)


def test_processing_tracker_generate_job_id_unique():
    """Verifies that different jobs produce different IDs."""
    # Different job names produce different IDs
    id1 = ProcessingTracker.generate_job_id("job1", "")
    id2 = ProcessingTracker.generate_job_id("job2", "")
    assert id1 != id2

    # Same job name with different specifiers produce different IDs
    id3 = ProcessingTracker.generate_job_id("process_plane", "plane_0")
    id4 = ProcessingTracker.generate_job_id("process_plane", "plane_1")
    assert id3 != id4


def test_processing_tracker_generate_job_id_without_specifier():
    """Verifies that generate_job_id works without a specifier."""
    job_name = "suite2p_processing"

    # With empty specifier
    id1 = ProcessingTracker.generate_job_id(job_name, "")
    # Without specifier (default)
    id2 = ProcessingTracker.generate_job_id(job_name)

    assert id1 == id2
    assert len(id1) == 16


def test_processing_tracker_initialize_jobs(tmp_path):
    """Verifies that initialize_jobs creates scheduled job entries and returns job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [
        ("job1", ""),
        ("job2", ""),
        ("job3", ""),
    ]

    job_ids = tracker.initialize_jobs(jobs=jobs)

    # Verifies returned job IDs
    assert len(job_ids) == 3
    for (job_name, specifier), job_id in zip(jobs, job_ids):
        assert job_id == ProcessingTracker.generate_job_id(job_name, specifier)

    # Reloads to verify persistence
    tracker._load_state()
    assert len(tracker.jobs) == 3
    for job_name, specifier in jobs:
        job_id = ProcessingTracker.generate_job_id(job_name, specifier)
        assert job_id in tracker.jobs
        assert tracker.jobs[job_id].job_name == job_name
        assert tracker.jobs[job_id].specifier == specifier
        assert tracker.jobs[job_id].status == ProcessingStatus.SCHEDULED
        assert tracker.jobs[job_id].executor_id is None


def test_processing_tracker_initialize_jobs_preserves_existing(tmp_path):
    """Verifies that initialize_jobs doesn't overwrite existing job entries."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [
        ("job1", ""),
        ("job2", ""),
    ]
    job_ids = [ProcessingTracker.generate_job_id(name, spec) for name, spec in jobs]

    # Initializes first time
    tracker.initialize_jobs(jobs=jobs)

    # Starts one job
    tracker.start_job(job_ids[0])

    # Reinitializes with the same jobs
    tracker.initialize_jobs(jobs=jobs)

    # Verifies the first job's status is preserved
    tracker._load_state()
    assert tracker.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED


def test_processing_tracker_find_jobs(tmp_path):
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

    # Searches by job name only
    matches = tracker.find_jobs(job_name="process_plane")
    assert len(matches) == 3

    # Searches by specifier only
    matches = tracker.find_jobs(specifier="plane_1")
    assert len(matches) == 1
    assert list(matches.values())[0] == ("process_plane", "plane_1")

    # Searches by partial job name
    matches = tracker.find_jobs(job_name="process")
    assert len(matches) == 3

    # Searches by partial specifier
    matches = tracker.find_jobs(specifier="plane")
    assert len(matches) == 3

    # Searches by both name and specifier
    matches = tracker.find_jobs(job_name="process_plane", specifier="plane_0")
    assert len(matches) == 1

    # No matches
    matches = tracker.find_jobs(job_name="nonexistent")
    assert len(matches) == 0


def test_processing_tracker_find_jobs_requires_argument(tmp_path):
    """Verifies that find_jobs raises ValueError when no arguments are provided."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    with pytest.raises(ValueError, match="At least one"):
        tracker.find_jobs()


def test_processing_tracker_start_job(tmp_path):
    """Verifies that start_job marks a job as running."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_id].executor_id is None


def test_processing_tracker_start_job_with_executor_id(tmp_path):
    """Verifies that start_job records the executor_id when provided."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id, executor_id="slurm-12345")

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_id].executor_id == "slurm-12345"


def test_processing_tracker_start_job_raises_for_unknown_job(tmp_path):
    """Verifies that start_job raises ValueError for unknown job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    unknown_job_id = "nonexistent_job_id"

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.start_job(unknown_job_id)


def test_processing_tracker_complete_job(tmp_path):
    """Verifies that complete_job marks a job as succeeded."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id)
    tracker.complete_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_fail_job(tmp_path):
    """Verifies that fail_job marks a job as failed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id)
    tracker.fail_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.FAILED


def test_processing_tracker_get_job_status(tmp_path):
    """Verifies that get_job_status returns the correct status."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    tracker.initialize_jobs(jobs=[("test_job", "")])

    # Checks scheduled status
    assert tracker.get_job_status(job_id) == ProcessingStatus.SCHEDULED

    # Starts and checks the running status
    tracker.start_job(job_id)
    assert tracker.get_job_status(job_id) == ProcessingStatus.RUNNING

    # Completes and checks succeeded status
    tracker.complete_job(job_id)
    assert tracker.get_job_status(job_id) == ProcessingStatus.SUCCEEDED


def test_processing_tracker_reset(tmp_path):
    """Verifies that reset clears all jobs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(name, spec) for name, spec in jobs]

    tracker.initialize_jobs(jobs=jobs)
    tracker.start_job(job_ids[0])

    # Resets
    tracker.reset()

    tracker._load_state()
    assert len(tracker.jobs) == 0


def test_processing_tracker_complete_property(tmp_path):
    """Verifies that the complete property returns True when all jobs succeed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(name, spec) for name, spec in jobs]

    tracker.initialize_jobs(jobs=jobs)
    assert not tracker.complete

    # Completes the first job
    tracker.start_job(job_ids[0])
    tracker.complete_job(job_ids[0])
    assert not tracker.complete

    # Completes the second job
    tracker.start_job(job_ids[1])
    tracker.complete_job(job_ids[1])
    assert tracker.complete


def test_processing_tracker_encountered_error_property(tmp_path):
    """Verifies that the encountered_error property returns True when any job fails."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "")]
    job_ids = [ProcessingTracker.generate_job_id(name, spec) for name, spec in jobs]

    tracker.initialize_jobs(jobs=jobs)
    assert not tracker.encountered_error

    # Completes the first job successfully
    tracker.start_job(job_ids[0])
    tracker.complete_job(job_ids[0])
    assert not tracker.encountered_error

    # Fails second job
    tracker.start_job(job_ids[1])
    tracker.fail_job(job_ids[1])
    assert tracker.encountered_error


def test_processing_tracker_concurrent_access(tmp_path):
    """Verifies that file locks prevent race conditions."""
    tracker_file = tmp_path / "tracker.yaml"

    # Simulates two processes
    tracker1 = ProcessingTracker(file_path=tracker_file)
    tracker2 = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")

    # Initializes from the first process
    tracker1.initialize_jobs(jobs=[("test_job", "")])

    # The second process can see the job
    assert tracker2.get_job_status(job_id) == ProcessingStatus.SCHEDULED

    # The first process starts the job
    tracker1.start_job(job_id)

    # The second process sees the update
    assert tracker2.get_job_status(job_id) == ProcessingStatus.RUNNING


def test_processing_tracker_yaml_serialization(tmp_path):
    """Verifies that the tracker state is properly serialized to YAML."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", "specifier")]
    job_ids = [ProcessingTracker.generate_job_id(name, spec) for name, spec in jobs]

    tracker.initialize_jobs(jobs=jobs)
    tracker.start_job(job_ids[0])

    # Creates a new instance and verifies it loads correctly
    tracker2 = ProcessingTracker(file_path=tracker_file)
    tracker2._load_state()

    assert len(tracker2.jobs) == 2
    assert tracker2.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert tracker2.jobs[job_ids[0]].job_name == "job1"
    assert tracker2.jobs[job_ids[0]].specifier == ""
    assert tracker2.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED
    assert tracker2.jobs[job_ids[1]].job_name == "job2"
    assert tracker2.jobs[job_ids[1]].specifier == "specifier"


def test_processing_status_enum_values():
    """Verifies all ProcessingStatus enumeration values."""
    assert ProcessingStatus.SCHEDULED == 0
    assert ProcessingStatus.RUNNING == 1
    assert ProcessingStatus.SUCCEEDED == 2
    assert ProcessingStatus.FAILED == 3


def test_job_state_defaults():
    """Verifies default JobState initialization."""
    job = JobState(job_name="test_job")
    assert job.job_name == "test_job"
    assert job.specifier == ""
    assert job.status == ProcessingStatus.SCHEDULED
    assert job.executor_id is None


def test_job_state_with_specifier():
    """Verifies JobState initialization with specifier."""
    job = JobState(job_name="process_plane", specifier="plane_0")
    assert job.job_name == "process_plane"
    assert job.specifier == "plane_0"
    assert job.status == ProcessingStatus.SCHEDULED


def test_job_state_with_executor_id():
    """Verifies JobState initialization with executor_id."""
    job = JobState(job_name="test_job", status=ProcessingStatus.RUNNING, executor_id="pid-42")
    assert job.job_name == "test_job"
    assert job.status == ProcessingStatus.RUNNING
    assert job.executor_id == "pid-42"


def test_job_state_new_fields():
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


def test_processing_tracker_timestamps(tmp_path):
    """Verifies that start_job and complete_job set timestamps."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")
    tracker.initialize_jobs(jobs=[("test_job", "")])

    # Starts the job and verifies started_at is set
    tracker.start_job(job_id)
    job_info = tracker.get_job_info(job_id)
    assert job_info.started_at is not None
    assert job_info.completed_at is None

    # Completes the job and verifies completed_at is set
    tracker.complete_job(job_id)
    job_info = tracker.get_job_info(job_id)
    assert job_info.completed_at is not None
    assert job_info.completed_at >= job_info.started_at


def test_processing_tracker_fail_job_with_error_message(tmp_path):
    """Verifies that fail_job records the error message."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("test_job", "")
    tracker.initialize_jobs(jobs=[("test_job", "")])
    tracker.start_job(job_id)
    tracker.fail_job(job_id, error_message="CUDA out of memory")

    job_info = tracker.get_job_info(job_id)
    assert job_info.status == ProcessingStatus.FAILED
    assert job_info.error_message == "CUDA out of memory"
    assert job_info.completed_at is not None


def test_processing_tracker_get_jobs_by_status(tmp_path):
    """Verifies that get_jobs_by_status returns correct job IDs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", ""), ("job4", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    # Initially all are scheduled
    scheduled = tracker.get_jobs_by_status(ProcessingStatus.SCHEDULED)
    assert len(scheduled) == 4

    # Starts two jobs
    tracker.start_job(job_ids[0])
    tracker.start_job(job_ids[1])
    running = tracker.get_jobs_by_status(ProcessingStatus.RUNNING)
    assert len(running) == 2
    assert job_ids[0] in running
    assert job_ids[1] in running

    # Completes one, fails one
    tracker.complete_job(job_ids[0])
    tracker.fail_job(job_ids[1])
    succeeded = tracker.get_jobs_by_status(ProcessingStatus.SUCCEEDED)
    failed = tracker.get_jobs_by_status(ProcessingStatus.FAILED)
    assert len(succeeded) == 1
    assert len(failed) == 1


def test_processing_tracker_get_summary(tmp_path):
    """Verifies that get_summary returns correct counts."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", ""), ("job4", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    # Initially all scheduled
    summary = tracker.get_summary()
    assert summary[ProcessingStatus.SCHEDULED] == 4
    assert summary[ProcessingStatus.RUNNING] == 0
    assert summary[ProcessingStatus.SUCCEEDED] == 0
    assert summary[ProcessingStatus.FAILED] == 0

    # Mixed states
    tracker.start_job(job_ids[0])
    tracker.start_job(job_ids[1])
    tracker.complete_job(job_ids[0])
    tracker.fail_job(job_ids[1])

    summary = tracker.get_summary()
    assert summary[ProcessingStatus.SCHEDULED] == 2
    assert summary[ProcessingStatus.RUNNING] == 0
    assert summary[ProcessingStatus.SUCCEEDED] == 1
    assert summary[ProcessingStatus.FAILED] == 1


def test_processing_tracker_get_job_info(tmp_path):
    """Verifies that get_job_info returns the full JobState."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    job_id = ProcessingTracker.generate_job_id("process_plane", "plane_0")
    tracker.initialize_jobs(jobs=[("process_plane", "plane_0")])
    tracker.start_job(job_id, executor_id="slurm-12345")

    job_info = tracker.get_job_info(job_id)
    assert job_info.job_name == "process_plane"
    assert job_info.specifier == "plane_0"
    assert job_info.status == ProcessingStatus.RUNNING
    assert job_info.executor_id == "slurm-12345"
    assert job_info.started_at is not None


def test_processing_tracker_get_job_info_raises_for_unknown(tmp_path):
    """Verifies that get_job_info raises ValueError for unknown job."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("test_job", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.get_job_info("nonexistent_id")


def test_processing_tracker_retry_failed_jobs(tmp_path):
    """Verifies that retry_failed_jobs resets failed jobs."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    jobs = [("job1", ""), ("job2", ""), ("job3", "")]
    job_ids = tracker.initialize_jobs(jobs=jobs)

    # Fails two jobs
    tracker.start_job(job_ids[0])
    tracker.start_job(job_ids[1])
    tracker.fail_job(job_ids[0], error_message="Error 1")
    tracker.fail_job(job_ids[1], error_message="Error 2")
    tracker.start_job(job_ids[2])
    tracker.complete_job(job_ids[2])

    # Retries failed jobs
    retried = tracker.retry_failed_jobs()
    assert len(retried) == 2
    assert job_ids[0] in retried
    assert job_ids[1] in retried

    # Verifies reset state
    job_info = tracker.get_job_info(job_ids[0])
    assert job_info.status == ProcessingStatus.SCHEDULED
    assert job_info.error_message is None
    assert job_info.started_at is None
    assert job_info.completed_at is None
    assert job_info.executor_id is None

    # Succeeded job should not be affected
    job_info = tracker.get_job_info(job_ids[2])
    assert job_info.status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_complete_job_invalid_id(tmp_path):
    """Verifies that complete_job raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.complete_job("invalid_job_id")


def test_processing_tracker_fail_job_invalid_id(tmp_path):
    """Verifies that fail_job raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.fail_job("invalid_job_id")


def test_processing_tracker_get_job_status_invalid_id(tmp_path):
    """Verifies that get_job_status raises ValueError for invalid job ID."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)
    tracker.initialize_jobs(jobs=[("job1", "")])

    with pytest.raises(ValueError, match="not configured to track"):
        tracker.get_job_status("invalid_job_id")


def test_processing_tracker_complete_property_empty_jobs(tmp_path):
    """Verifies that complete property returns False for empty tracker."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    # Saves empty state.
    tracker._save_state()

    assert not tracker.complete
