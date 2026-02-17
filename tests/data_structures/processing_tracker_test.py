"""Contains tests for the ProcessingTracker, ProcessingStatus, and JobState classes."""

from pathlib import Path

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
    source_path = Path("/data/project/animal/session")
    job_name = "suite2p_processing"

    # Generates the same ID multiple times
    id1 = ProcessingTracker.generate_job_id(source_path, job_name)
    id2 = ProcessingTracker.generate_job_id(source_path, job_name)

    # Should be consistent
    assert id1 == id2
    # Should be a hexadecimal string
    assert len(id1) == 16
    assert all(c in "0123456789abcdef" for c in id1)


def test_processing_tracker_generate_job_id_unique():
    """Verifies that different jobs produce different IDs."""
    source_path = Path("/data/project/animal/session")

    id1 = ProcessingTracker.generate_job_id(source_path, "job1")
    id2 = ProcessingTracker.generate_job_id(source_path, "job2")

    assert id1 != id2


def test_processing_tracker_initialize_jobs(tmp_path):
    """Verifies that initialize_jobs creates scheduled job entries."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
        ProcessingTracker.generate_job_id(source_path, "job3"),
    ]

    tracker.initialize_jobs(job_ids=job_ids)

    # Reloads to verify persistence
    tracker._load_state()
    assert len(tracker.jobs) == 3
    for job_id in job_ids:
        assert job_id in tracker.jobs
        assert tracker.jobs[job_id].status == ProcessingStatus.SCHEDULED
        assert tracker.jobs[job_id].executor_id is None


def test_processing_tracker_initialize_jobs_preserves_existing(tmp_path):
    """Verifies that initialize_jobs doesn't overwrite existing job entries."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
    ]

    # Initializes first time
    tracker.initialize_jobs(job_ids=job_ids)

    # Starts one job
    tracker.start_job(job_ids[0])

    # Reinitializes with the same jobs
    tracker.initialize_jobs(job_ids=job_ids)

    # Verifies the first job's status is preserved
    tracker._load_state()
    assert tracker.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED


def test_processing_tracker_start_job(tmp_path):
    """Verifies that start_job marks a job as running."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    tracker.initialize_jobs(job_ids=[job_id])
    tracker.start_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.RUNNING
    assert tracker.jobs[job_id].executor_id is None


def test_processing_tracker_start_job_with_executor_id(tmp_path):
    """Verifies that start_job records the executor_id when provided."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    tracker.initialize_jobs(job_ids=[job_id])
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

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    tracker.initialize_jobs(job_ids=[job_id])
    tracker.start_job(job_id)
    tracker.complete_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.SUCCEEDED


def test_processing_tracker_fail_job(tmp_path):
    """Verifies that fail_job marks a job as failed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    tracker.initialize_jobs(job_ids=[job_id])
    tracker.start_job(job_id)
    tracker.fail_job(job_id)

    tracker._load_state()
    assert tracker.jobs[job_id].status == ProcessingStatus.FAILED


def test_processing_tracker_get_job_status(tmp_path):
    """Verifies that get_job_status returns the correct status."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    tracker.initialize_jobs(job_ids=[job_id])

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

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
    ]

    tracker.initialize_jobs(job_ids=job_ids)
    tracker.start_job(job_ids[0])

    # Resets
    tracker.reset()

    tracker._load_state()
    assert len(tracker.jobs) == 0


def test_processing_tracker_complete_property(tmp_path):
    """Verifies that the complete property returns True when all jobs succeed."""
    tracker_file = tmp_path / "tracker.yaml"
    tracker = ProcessingTracker(file_path=tracker_file)

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
    ]

    tracker.initialize_jobs(job_ids=job_ids)
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

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
    ]

    tracker.initialize_jobs(job_ids=job_ids)
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

    source_path = Path("/data/session")
    job_id = ProcessingTracker.generate_job_id(source_path, "test_job")

    # Initializes from the first process
    tracker1.initialize_jobs(job_ids=[job_id])

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

    source_path = Path("/data/session")
    job_ids = [
        ProcessingTracker.generate_job_id(source_path, "job1"),
        ProcessingTracker.generate_job_id(source_path, "job2"),
    ]

    tracker.initialize_jobs(job_ids=job_ids)
    tracker.start_job(job_ids[0])

    # Creates a new instance and verifies it loads correctly
    tracker2 = ProcessingTracker(file_path=tracker_file)
    tracker2._load_state()

    assert len(tracker2.jobs) == 2
    assert tracker2.jobs[job_ids[0]].status == ProcessingStatus.RUNNING
    assert tracker2.jobs[job_ids[1]].status == ProcessingStatus.SCHEDULED


def test_processing_status_enum_values():
    """Verifies all ProcessingStatus enumeration values."""
    assert ProcessingStatus.SCHEDULED == 0
    assert ProcessingStatus.RUNNING == 1
    assert ProcessingStatus.SUCCEEDED == 2
    assert ProcessingStatus.FAILED == 3


def test_job_state_defaults():
    """Verifies default JobState initialization."""
    job = JobState()
    assert job.status == ProcessingStatus.SCHEDULED
    assert job.executor_id is None


def test_job_state_with_executor_id():
    """Verifies JobState initialization with executor_id."""
    job = JobState(status=ProcessingStatus.RUNNING, executor_id="pid-42")
    assert job.status == ProcessingStatus.RUNNING
    assert job.executor_id == "pid-42"
