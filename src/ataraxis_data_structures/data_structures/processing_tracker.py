"""Provides assets for running data processing pipelines and tracking their progress."""

from enum import IntEnum
from pathlib import Path
from dataclasses import field, dataclass

import xxhash
from filelock import FileLock
from ataraxis_time import get_timestamp, TimestampFormats, TimestampPrecisions
from ataraxis_base_utilities import console, LogLevel

from .yaml_config import YamlConfig


class ProcessingStatus(IntEnum):
    """Defines the status codes used by the ProcessingTracker instances to communicate the runtime state of each
    job making up the managed data processing pipeline.
    """

    SCHEDULED = 0
    """Indicates the job is scheduled for execution."""
    RUNNING = 1
    """Indicates the job is currently being executed."""
    SUCCEEDED = 2
    """Indicates the job has been completed successfully."""
    FAILED = 3
    """Indicates the job encountered a runtime error and was not completed."""


@dataclass
class JobState:
    """Stores the metadata and the current runtime status of a single job in the processing pipeline."""

    job_name: str
    """The descriptive name of the job."""
    specifier: str = ""
    """An optional specifier that differentiates instances of the same job, for example, when running the same job
    over multiple batches of data."""
    status: ProcessingStatus = ProcessingStatus.SCHEDULED
    """The current status of the job."""
    executor_id: str | None = None
    """An optional identifier for the executor running the job (e.g. a SLURM job ID, a process PID, or any
    user-defined string)."""
    error_message: str | None = None
    """An optional error message describing why the job failed."""
    started_at: int | None = None
    """The UTC timestamp (microsecond-precision epoch) when the job started running."""
    completed_at: int | None = None
    """The UTC timestamp (microsecond-precision epoch) when the job completed (succeeded or failed)."""


@dataclass
class ProcessingTracker(YamlConfig):
    """Tracks the state of a data processing pipeline and provides tools for communicating this state between multiple
    processes and host-machines.

    Notes:
        All modifications to the tracker file require the acquisition of the .lock file, which ensures exclusive
        access to the tracker's data, allowing multiple independent processes (jobs) to safely work with the same
        tracker file.
    """

    file_path: Path
    """The path to the .YAML file used to cache the tracker's data on disk."""
    jobs: dict[str, JobState] = field(default_factory=dict)
    """Maps the unique identifiers of the jobs that make up the processing pipeline to their current state and
    metadata."""
    lock_path: str = field(init=False)
    """The path to the .LOCK file used to ensure thread-safe access to the tracker's data."""

    def __post_init__(self) -> None:
        """Resolves the .LOCK file for the managed tracker .YAML file."""
        # Generates the .lock file path for the target tracker .yaml file. Skips if file_path is empty (used during
        # serialization to avoid storing instance-specific paths).
        if self.file_path is not None and self.file_path.parts:
            self.lock_path = str(self.file_path.with_suffix(self.file_path.suffix + ".lock"))
        else:
            self.lock_path = ""

    @staticmethod
    def generate_job_id(job_name: str, specifier: str = "") -> str:
        """Generates a unique hexadecimal job identifier based on the job's name and optional specifier using the
        xxHash64 checksum generator.

        Args:
            job_name: The descriptive name for the processing job (e.g., 'process_data').
            specifier: An optional specifier that differentiates instances of the same job (e.g., 'batch_101').

        Returns:
            The unique hexadecimal identifier for the target job.
        """
        # Combines job name and specifier into a single string for hashing
        combined = f"{job_name}:{specifier}" if specifier else job_name
        # Generates and returns the xxHash64 hash
        return xxhash.xxh64(combined.encode("utf-8")).hexdigest()

    def _load_state(self) -> None:
        """Reads the processing pipeline's runtime state from the cached .YAML file."""
        if self.file_path.exists():
            loaded = ProcessingTracker.from_yaml(self.file_path)
            self.jobs = loaded.jobs
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Caches the current processing state stored inside the instance's attributes as a .YAML file."""
        # Temporarily sets file_path and lock_path to empty values to avoid serializing instance-specific paths.
        # YamlConfig's _serialize_value() automatically handles Enum -> value conversion.
        temp_file_path, temp_lock_path = self.file_path, self.lock_path
        try:
            self.file_path = Path()
            self.lock_path = ""
            self.to_yaml(file_path=temp_file_path)
        finally:
            self.file_path, self.lock_path = temp_file_path, temp_lock_path

    def initialize_jobs(self, jobs: list[tuple[str, str]]) -> list[str]:
        """Configures the tracker with the list of one or more jobs to be executed during the pipeline's runtime.

        Notes:
            If the job already has a section in the tracker, this method does not duplicate or modify the existing
            job entry. Use the reset() method to clear all cached job states.

        Args:
            jobs: A list of (job_name, specifier) tuples defining the jobs to track. Each tuple contains the
                descriptive job name and an optional specifier string. Use an empty string for jobs without a
                specifier.

        Returns:
            A list of job IDs corresponding to the input jobs.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            job_ids = []
            for job_name, specifier in jobs:
                job_id = self.generate_job_id(job_name, specifier)
                if job_id not in self.jobs:
                    self.jobs[job_id] = JobState(job_name=job_name, specifier=specifier)
                else:
                    console.echo(
                        message=f"Job '{job_name}' with specifier '{specifier}' (ID: {job_id}) already exists in the "
                        f"tracker. Skipping duplicate entry.",
                        level=LogLevel.WARNING,
                    )
                job_ids.append(job_id)

            self._save_state()
            return job_ids

    def find_jobs(
        self, job_name: str | None = None, specifier: str | None = None
    ) -> dict[str, tuple[str, str]]:
        """Searches for jobs matching the given name and/or specifier patterns.

        Supports partial matching (substring search) on job names and specifiers. If both parameters are provided,
        jobs must match both patterns.

        Args:
            job_name: A substring to match against job names. If None, matches any job name.
            specifier: A substring to match against specifiers. If None, matches any specifier.

        Returns:
            A dictionary mapping matching job IDs to (job_name, specifier) tuples.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If both job_name and specifier are None.
        """
        if job_name is None and specifier is None:
            message = "At least one of 'job_name' or 'specifier' must be provided for searching."
            console.error(message=message, error=ValueError)

        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            matches: dict[str, tuple[str, str]] = {}
            for job_id, job_state in self.jobs.items():
                name_match = job_name is None or job_name in job_state.job_name
                spec_match = specifier is None or specifier in job_state.specifier
                if name_match and spec_match:
                    matches[job_id] = (job_state.job_name, job_state.specifier)

            return matches

    def start_job(self, job_id: str, executor_id: str | None = None) -> None:
        """Marks the target job as running and optionally records the executor identifier.

        Args:
            job_id: The unique identifier of the job to mark as started.
            executor_id: An optional identifier for the executor running the job (e.g. a SLURM job ID, a process PID,
                or any user-defined string).

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            # Loads tracker state from the .yaml file
            self._load_state()

            # Verifies that the tracker is configured to track the specified job
            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            # Updates job status, records the executor identifier, and sets the start timestamp
            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.RUNNING
            job_info.executor_id = executor_id
            job_info.started_at = get_timestamp(
                output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND
            )

            self._save_state()

    def complete_job(self, job_id: str) -> None:
        """Marks a target job as successfully completed.

        Args:
            job_id: The unique identifier of the job to mark as complete.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            # Loads tracker state from the .yaml file
            self._load_state()

            # Verifies that the tracker is configured to track the specified job
            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            # Updates the job's status and sets the completion timestamp
            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.SUCCEEDED
            job_info.completed_at = get_timestamp(
                output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND
            )

            self._save_state()

    def fail_job(self, job_id: str, error_message: str | None = None) -> None:
        """Marks the target job as failed.

        Args:
            job_id: The unique identifier of the job to mark as failed.
            error_message: An optional error message describing why the job failed.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            # Loads tracker state from the .yaml file
            self._load_state()

            # Verifies that the tracker is configured to track the specified job
            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            # Updates the job's status, error message, and completion timestamp
            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.FAILED
            job_info.error_message = error_message
            job_info.completed_at = get_timestamp(
                output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND
            )

            self._save_state()

    def get_job_status(self, job_id: str) -> ProcessingStatus:
        """Queries the current runtime status of the target job.

        Args:
            job_id: The unique identifier of the job for which to query the runtime status.

        Returns:
            The current runtime status of the job.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            # Verifies that the tracker is configured to track the specified job
            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            return self.jobs[job_id].status

    def reset(self) -> None:
        """Resets the tracker file to the default state."""
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            # Loads tracker state from the .yaml file.
            self._load_state()

            # Resets the tracker file to the default state.
            self.jobs.clear()
            self._save_state()

    @property
    def complete(self) -> bool:
        """Returns True if the tracked processing pipeline has been completed successfully.

        Notes:
            The pipeline is considered complete if all jobs have been marked as succeeded.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            if not self.jobs:
                return False
            return all(job.status == ProcessingStatus.SUCCEEDED for job in self.jobs.values())

    @property
    def encountered_error(self) -> bool:
        """Returns True if the tracked processing pipeline has been terminated due to a runtime error.

        Notes:
            The pipeline is considered to have encountered an error if any job has been marked as failed.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            return any(job.status == ProcessingStatus.FAILED for job in self.jobs.values())

    def get_jobs_by_status(self, status: ProcessingStatus | str) -> list[str]:
        """Returns all job IDs that have the specified status.

        Args:
            status: The status to filter jobs by.

        Returns:
            A list of job IDs with the specified status.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            return [job_id for job_id, job_state in self.jobs.items() if job_state.status == ProcessingStatus(status)]

    def get_summary(self) -> dict[ProcessingStatus, int]:
        """Returns a summary of job counts by status.

        Returns:
            A dictionary mapping each ProcessingStatus to the count of jobs with that status.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            summary: dict[ProcessingStatus, int] = {status: 0 for status in ProcessingStatus}
            for job_state in self.jobs.values():
                summary[job_state.status] += 1
            return summary

    def get_job_info(self, job_id: str) -> JobState:
        """Returns the full JobState object for the specified job.

        Args:
            job_id: The unique identifier of the job to query.

        Returns:
            The JobState object containing all metadata for the job.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            return self.jobs[job_id]

    def retry_failed_jobs(self) -> list[str]:
        """Resets all failed jobs back to SCHEDULED status for retry.

        This clears the error_message, started_at, and completed_at fields for each failed job.

        Returns:
            A list of job IDs that were reset for retry.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            retried_jobs = []
            for job_id, job_state in self.jobs.items():
                if job_state.status == ProcessingStatus.FAILED:
                    job_state.status = ProcessingStatus.SCHEDULED
                    job_state.error_message = None
                    job_state.started_at = None
                    job_state.completed_at = None
                    job_state.executor_id = None
                    retried_jobs.append(job_id)

            self._save_state()
            return retried_jobs
