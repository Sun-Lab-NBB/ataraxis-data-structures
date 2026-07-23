"""Provides assets for running data processing pipelines and tracking their progress."""

import os
from enum import IntEnum
from pathlib import Path
from dataclasses import field, replace, dataclass

import xxhash
from filelock import FileLock
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from .yaml_config import YamlConfig

_SCHEDULER_EXECUTOR_SOURCES: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    # SLURM sets SLURM_JOB_ID on current versions and SLURM_JOBID on older ones, both naming the same allocation.
    ("slurm", ("SLURM_JOB_ID", "SLURM_JOBID"), None),
    # PBS, Torque, and OpenPBS expose the job ID under PBS_JOBID.
    ("pbs", ("PBS_JOBID",), None),
    # IBM Spectrum LSF.
    ("lsf", ("LSB_JOBID",), None),
    # OAR.
    ("oar", ("OAR_JOB_ID",), None),
    # Grid Engine (SGE and its Altair and Univa descendants) names the job ID with the generic JOB_ID variable, so
    # it is accepted only when SGE_ROOT corroborates that the process runs under a Grid Engine allocation.
    ("sge", ("JOB_ID",), "SGE_ROOT"),
    # Microsoft HPC Pack, the on-premise Windows HPC scheduler.
    ("hpcpack", ("CCP_JOBID",), None),
    # Azure Batch, across Linux and Windows compute nodes.
    ("azurebatch", ("AZ_BATCH_JOB_ID",), None),
    # AWS Batch.
    ("awsbatch", ("AWS_BATCH_JOB_ID",), None),
)
"""Ordered job-scheduler detection sources consulted when resolving an executor identifier from the environment.

Each entry pairs a scheme label with the environment variables that carry the scheduler's job ID, in priority order,
and an optional corroborating variable that must also be present for the source to apply. The scheme label is
recorded alongside the ID so a downstream consumer can select the matching liveness query (``sacct`` for SLURM,
``qstat`` for PBS, and so on). This module only records the identifier, the per-scheme reconciliation lives with the
consumer that owns the scheduler binding.
"""


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


@dataclass(slots=True)
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
    """The path to the .LOCK file used to ensure process-safe access to the tracker's data."""

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
        combined = f"{job_name}:{specifier}" if specifier else job_name
        return xxhash.xxh64(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_executor_id() -> str:
        """Resolves the identifier of the executor running a job from the runtime environment.

        Consults the recognized job schedulers in ``_SCHEDULER_EXECUTOR_SOURCES`` in priority order and returns the
        first scheduler job ID found, tagged with its scheme as ``"<scheme>:<id>"``. The tag lets a stale tracker
        entry be correlated with the scheduler's own record of the job. Falls back to ``"pid:<process id>"`` when the
        process runs under no recognized scheduler, so a locally executed job still records a meaningful executor
        identifier. The scheme prefix lets a consumer select the liveness query that matches the executor.

        Returns:
            The scheme-tagged scheduler job ID when the process runs under a recognized scheduler, otherwise the
            scheme-tagged process ID.
        """
        for scheme, id_variables, corroborating_variable in _SCHEDULER_EXECUTOR_SOURCES:
            if corroborating_variable is not None and corroborating_variable not in os.environ:
                continue
            for id_variable in id_variables:
                job_id = os.environ.get(id_variable)
                if job_id:
                    return f"{scheme}:{job_id}"
        return f"pid:{os.getpid()}"

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
            If the job already has a section in the tracker, this method emits a warning and does not duplicate or
            modify the existing job entry. Use the reset() method to clear all cached job states.

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
                job_id = self.generate_job_id(job_name=job_name, specifier=specifier)
                if job_id not in self.jobs:
                    self.jobs[job_id] = JobState(job_name=job_name, specifier=specifier)
                else:
                    # Temporarily enables console output to ensure the warning is visible, then restores previous state.
                    was_enabled = console.enabled
                    if not was_enabled:
                        console.enable()
                    console.echo(
                        message=f"Job '{job_name}' with specifier '{specifier}' (ID: {job_id}) already exists in the "
                        f"tracker. Skipping duplicate entry.",
                        level=LogLevel.WARNING,
                    )
                    if not was_enabled:
                        console.disable()
                job_ids.append(job_id)

            self._save_state()
            return job_ids

    def align_jobs(self, jobs: list[tuple[str, str]], universe: list[tuple[str, str]] | None = None) -> list[str]:
        """Aligns the tracker's job registry with the jobs requested for the current pipeline invocation.

        Notes:
            Foreign entries are detected against ``universe``, the full set of jobs the pipeline could produce,
            rather than against the requested subset. That distinction lets an invocation run part of a pipeline
            while its siblings keep their recorded state. A registry holding entries outside the universe means the
            pipeline's own definition has changed since the tracker was written, so the registry is rebuilt and the
            discarded IDs are reported through a warning.

            Otherwise, the method additively registers any requested job the registry is missing, and is a no-op
            when every requested job is already present.

        Args:
            jobs: The (job_name, specifier) tuples the current invocation intends to execute.
            universe: The (job_name, specifier) tuples enumerating every job the pipeline could produce for its
                current definition, used only to detect foreign entries. Defaults to ``jobs``, which is correct for
                a pipeline whose requested set is always its full universe.

        Returns:
            A list of job IDs corresponding to the requested jobs.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        resolved_universe = jobs if universe is None else universe
        universe_ids = {
            self.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in resolved_universe
        }
        requested = [
            (self.generate_job_id(job_name=job_name, specifier=specifier), job_name, specifier)
            for job_name, specifier in jobs
        ]

        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            foreign_ids = sorted(set(self.jobs) - universe_ids)
            if foreign_ids:
                # Temporarily enables console output to ensure the warning is visible, then restores previous state.
                was_enabled = console.enabled
                if not was_enabled:
                    console.enable()
                console.echo(
                    message=(
                        f"The processing tracker at '{self.file_path}' contains {len(foreign_ids)} job entries that "
                        f"are not part of the current job universe. Rebuilding the tracker to match the requested "
                        f"jobs. Discarded job IDs: {foreign_ids}."
                    ),
                    level=LogLevel.WARNING,
                )
                if not was_enabled:
                    console.disable()
                self.jobs.clear()

            # Registers the requested jobs that are absent, preserving the state of every job already tracked.
            for job_id, job_name, specifier in requested:
                if job_id not in self.jobs:
                    self.jobs[job_id] = JobState(job_name=job_name, specifier=specifier)

            self._save_state()
            return [job_id for job_id, _, _ in requested]

    def snapshot(self) -> dict[str, JobState]:
        """Returns a point-in-time copy of the tracker's complete job registry.

        Notes:
            Reads the whole registry under a single lock acquisition, so the returned states are consistent with
            each other. This is the method to use when reporting on every tracked job at once.

            The returned states are copies, so mutating them does not affect the tracker. Use the dedicated
            mutators to change job state.

            A tracker file that does not exist yields an empty registry and is left uncreated, so probing a
            pipeline that has never run leaves its output directory unchanged.

        Returns:
            A dictionary mapping every tracked job ID to a copy of its ``JobState``, or an empty dictionary when
            the tracker file does not exist.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        if not self.file_path.is_file():
            return {}

        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            # Copies each state so the caller cannot mutate the instance's registry. Every JobState field is an
            # immutable scalar, so a per-entry replace() is a complete copy and is cheaper than a deep copy.
            return {job_id: replace(job_state) for job_id, job_state in self.jobs.items()}

    def find_jobs(self, job_name: str | None = None, specifier: str | None = None) -> dict[str, tuple[str, str]]:
        """Searches for jobs matching the given name and/or specifier patterns.

        Supports partial matching (substring search) on job names and specifiers. If both parameters are provided,
        jobs must match both patterns.

        Args:
            job_name: A substring to match against job names. If None, matches any job name.
            specifier: A substring to match against specifiers. If None, matches any specifier.

        Returns:
            A dictionary mapping matching job IDs to (job_name, specifier) tuples. Calling the method without
            arguments matches every tracked job.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
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
        """Marks the target job as running and records the identifier of the executor running it.

        Args:
            job_id: The unique identifier of the job to mark as started.
            executor_id: An optional explicit identifier for the executor running the job. When None (default), the
                identifier is resolved automatically from the runtime environment, preferring a recognized job
                scheduler's job ID and falling back to the process ID, each tagged with its scheme.

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

            # Resolves the executor identifier from the runtime environment (SLURM job ID, falling back to the
            # process ID) when the caller does not provide one explicitly.
            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.RUNNING
            job_info.executor_id = executor_id if executor_id is not None else self._resolve_executor_id()
            job_info.started_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)
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
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.SUCCEEDED
            job_info.completed_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)
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
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job with ID "
                    f"'{job_id}'. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.FAILED
            job_info.error_message = error_message
            job_info.completed_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)
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
            self._load_state()

            self.jobs.clear()
            self._save_state()

    @property
    def complete(self) -> bool:
        """Returns True when the tracked pipeline has jobs and all of them have been marked as succeeded."""
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            if not self.jobs:
                return False
            return all(job.status == ProcessingStatus.SUCCEEDED for job in self.jobs.values())

    @property
    def encountered_error(self) -> bool:
        """Returns True when any of the tracked pipeline's jobs has been marked as failed."""
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            return any(job.status == ProcessingStatus.FAILED for job in self.jobs.values())

    def get_jobs_by_status(self, status: ProcessingStatus | str) -> list[str]:
        """Returns all job IDs that have the specified status.

        Args:
            status: The status to match, given as a ProcessingStatus member or its member name string.

        Returns:
            A list of job IDs with the specified status.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            KeyError: If status is a string that does not name a valid ProcessingStatus member.
        """
        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()
            target_status = ProcessingStatus[status] if isinstance(status, str) else status
            return [job_id for job_id, job_state in self.jobs.items() if job_state.status == target_status]

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
            summary: dict[ProcessingStatus, int] = dict.fromkeys(ProcessingStatus, 0)
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

    def reset_jobs(self, job_ids: list[str]) -> list[str]:
        """Resets the specified jobs back to SCHEDULED status, leaving every other job's state untouched.

        Clears the error_message, started_at, completed_at, and executor_id fields of each targeted job.

        Notes:
            Every job outside ``job_ids`` keeps its recorded status, executor, and timestamps, so this is the method
            to use when re-running part of a pipeline.

            A job ID the tracker does not know means the caller's view of the registry disagrees with the registry
            itself, so the method rejects the whole request. Changing which jobs a tracker holds is the job of
            ``align_jobs``. The membership check completes before any job is modified, so a rejected request leaves
            the tracker untouched.

        Args:
            job_ids: The unique identifiers of the jobs to reset.

        Returns:
            A list of the reset job IDs, in registry order.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If any of the specified job IDs is not found in the managed tracker file.
        """
        targeted = set(job_ids)

        lock = FileLock(self.lock_path)
        with lock.acquire(timeout=10.0):
            self._load_state()

            unknown_ids = sorted(targeted - set(self.jobs))
            if unknown_ids:
                message = (
                    f"The ProcessingTracker instance is not configured to track the state of the job(s) with ID(s): "
                    f"{', '.join(unknown_ids)}. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            reset_ids = [job_id for job_id in self.jobs if job_id in targeted]
            for job_id in reset_ids:
                job_state = self.jobs[job_id]
                job_state.status = ProcessingStatus.SCHEDULED
                job_state.error_message = None
                job_state.started_at = None
                job_state.completed_at = None
                job_state.executor_id = None

            self._save_state()
            return reset_ids

    def retry_failed_jobs(self) -> list[str]:
        """Resets all failed jobs back to SCHEDULED status for retry.

        Clears the error_message, started_at, completed_at, and executor_id fields for each failed job.

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
