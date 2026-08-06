"""Provides assets for running data processing pipelines and tracking their progress."""

import os
from enum import IntEnum, StrEnum
from typing import Any, Self
from pathlib import Path
from contextlib import contextmanager
from dataclasses import field, replace, dataclass
from collections.abc import Mapping, Iterator

import xxhash
from filelock import FileLock
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from ..processing import discover_marker_files
from .yaml_config import YAML_EXCLUDE_METADATA, YamlConfig

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
``qstat`` for PBS, and so on). This module only records the identifier. The per-scheme reconciliation lives with the
consumer that owns the scheduler binding.
"""

_LOCK_ACQUISITION_TIMEOUT: float = 10.0
"""The maximum time, in seconds, to wait for the tracker's .LOCK file before aborting the operation."""


class ProcessingStatus(IntEnum):
    """Defines the status codes used by the ``ProcessingTracker`` instances to communicate the runtime state of each
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


class TrackerStatus(StrEnum):
    """Defines the high-level progress labels that summarize the job registry of a ``ProcessingTracker`` instance.

    Notes:
        A label describes the pipeline the tracker follows, while a ``ProcessingStatus`` member describes one job
        inside that pipeline.
    """

    NOT_STARTED = "not_started"
    """Indicates every tracked job is still scheduled."""
    IN_PROGRESS = "in_progress"
    """Indicates the tracked jobs have mixed outcomes with none running and none failed, which also covers a tracker
    holding no jobs at all."""
    PROCESSING = "processing"
    """Indicates at least one tracked job is currently running."""
    COMPLETED = "completed"
    """Indicates every tracked job succeeded."""
    FAILED = "failed"
    """Indicates at least one tracked job failed."""


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
    """An optional identifier for the executor running the job (e.g., a SLURM job ID, a process PID, or any
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
    processes and host machines.

    Notes:
        All modifications to the tracker file require the acquisition of the .lock file, which ensures exclusive
        access to the tracker's data, allowing multiple independent processes (jobs) to safely work with the same
        tracker file.
    """

    file_path: Path = field(metadata=YAML_EXCLUDE_METADATA)
    """The path to the .YAML file used to cache the tracker's data on disk. Excluded from the serialized document,
    since it records where the tracker lives rather than the pipeline state the tracker holds."""
    jobs: dict[str, JobState] = field(default_factory=dict)
    """Maps the unique identifiers of the jobs that make up the processing pipeline to their current state and
    metadata."""
    lock_path: str = field(init=False, metadata=YAML_EXCLUDE_METADATA)
    """The path to the .LOCK file used to ensure process-safe access to the tracker's data. Excluded from the
    serialized document, since it is derived from the file path."""

    def __post_init__(self) -> None:
        """Resolves the .LOCK file for the managed tracker .YAML file."""
        self.lock_path = str(self.file_path.with_suffix(self.file_path.suffix + ".lock"))

    @classmethod
    def restore_excluded_fields(cls, data: dict[Any, Any], file_path: Path) -> dict[Any, Any]:
        """Reattaches the reconstructed tracker to the file it was read from.

        Notes:
            This method overrides the ``YamlConfig`` implementation and runs only as part of that class's
            deserialization machinery, which calls it from ``from_yaml()`` between reading the document and building
            the instance. Nothing calls it directly.

            The tracker marks both of its path fields with ``YAML_EXCLUDE_METADATA``, so the document it writes
            holds the job registry alone and offers the constructor no path to take. Supplying the path here keeps
            every instance ``from_yaml()`` returns bound to a real file.

        Args:
            data: The top-level mapping read from the tracker .yaml file.
            file_path: The path the mapping was read from.

        Returns:
            The mapping extended with the tracker's own file path.
        """
        return {**data, "file_path": file_path}

    @classmethod
    def discover(cls, root_directory: Path, tracker_name: str) -> list[Self]:
        """Discovers every processing tracker with the target filename stored anywhere under the root directory.

        Notes:
            Each returned instance is bound to the file it was discovered at and holds no job state until one of its
            readers loads that file, so surveying a large tree costs one traversal rather than one read per tracker.

        Args:
            root_directory: The root directory whose tree is searched.
            tracker_name: The exact filename every discovered tracker .yaml file carries.

        Returns:
            A tracker bound to every matching file found anywhere under the root directory, sorted by file path.

        Raises:
            OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
                beneath it cannot be read.
        """
        return [
            cls(file_path=tracker_path)
            for tracker_path in discover_marker_files(directory=root_directory, marker_name=tracker_name)
        ]

    @staticmethod
    def generate_job_id(job_name: str, specifier: str = "") -> str:
        """Generates a unique hexadecimal job identifier based on the job's name and optional specifier using the
        xxHash64 checksum generator.

        Notes:
            A colon joins the two components inside the hashed string, so neither component may contain one. Were a
            colon allowed, the pairs ('data:batch', '') and ('data', 'batch') would join to the same string and
            therefore to the same identifier, collapsing two declared jobs onto one registry entry.

        Args:
            job_name: The descriptive name for the processing job (e.g., 'process_data'). Cannot contain a colon.
            specifier: An optional specifier that differentiates instances of the same job (e.g., 'batch_101').
                Cannot contain a colon.

        Returns:
            The unique hexadecimal identifier for the target job.

        Raises:
            ValueError: If the job name or the specifier contains a colon.
        """
        if ":" in job_name or ":" in specifier:
            message = (
                f"Unable to generate the identifier for the job '{job_name}' with the specifier '{specifier}'. The "
                f"job name and the specifier must not contain the ':' character, as it joins them inside the hashed "
                f"identifier string."
            )
            console.error(message=message, error=ValueError)

        combined = f"{job_name}:{specifier}" if specifier else job_name
        return xxhash.xxh64(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def resolve_status(summary: Mapping[str, int]) -> TrackerStatus:
        """Resolves the high-level progress label that a tracker summary's job counts describe.

        Notes:
            Applies a fixed priority. A summary counting a failed job resolves to FAILED whatever else it counts, one
            whose every job succeeded resolves to COMPLETED, one counting a running job resolves to PROCESSING, and
            one whose every job is still scheduled resolves to NOT_STARTED. Every remaining case resolves to
            IN_PROGRESS, which covers a summary counting no jobs at all, since the COMPLETED and NOT_STARTED branches
            each require at least one counted job.

            Counts aggregated across several trackers resolve the same way, so a caller reporting on a directory of
            trackers labels the group with the same priority it labels each member.

        Args:
            summary: The job counts to resolve, carrying the 'total', 'succeeded', 'failed', 'running', and
                'scheduled' keys that ``summarize()`` produces. An absent key counts as zero.

        Returns:
            The label matching the highest-priority condition the counts satisfy.
        """
        total = summary.get("total", 0)
        if summary.get("failed", 0) > 0:
            return TrackerStatus.FAILED
        if total > 0 and summary.get("succeeded", 0) == total:
            return TrackerStatus.COMPLETED
        if summary.get("running", 0) > 0:
            return TrackerStatus.PROCESSING
        if total > 0 and summary.get("scheduled", 0) == total:
            return TrackerStatus.NOT_STARTED
        return TrackerStatus.IN_PROGRESS

    def initialize_jobs(self, jobs: list[tuple[str, str]]) -> list[str]:
        """Configures the tracker with the list of one or more jobs to be executed during the pipeline's runtime.

        Notes:
            If the job already has a section in the tracker, this method emits a warning and does not duplicate or
            modify the existing job entry. Use the ``reset()`` method to clear all cached job states.

        Args:
            jobs: A list of (job_name, specifier) tuples defining the jobs to track. Each tuple contains the
                descriptive job name and an optional specifier string. Use an empty string for jobs without a
                specifier.

        Returns:
            A list of job IDs corresponding to the input jobs.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If any job name or specifier contains a colon.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            job_ids = []
            for job_name, specifier in jobs:
                job_id = self.generate_job_id(job_name=job_name, specifier=specifier)
                if job_id not in self.jobs:
                    self.jobs[job_id] = JobState(job_name=job_name, specifier=specifier)
                else:
                    message = (
                        f"Job '{job_name}' with specifier '{specifier}' (ID: {job_id}) already exists in the tracker. "
                        f"Skipping duplicate entry."
                    )
                    with console.temporarily_enabled():
                        console.echo(message=message, level=LogLevel.WARNING)
                job_ids.append(job_id)

            self._save_state()
            return job_ids

    def align_jobs(self, jobs: list[tuple[str, str]], universe: list[tuple[str, str]] | None = None) -> list[str]:
        """Aligns the tracker's job registry with the jobs requested for the current pipeline invocation.

        Notes:
            Foreign entries are detected against ``universe``, the full set of jobs the pipeline could produce,
            rather than against the requested subset. That distinction lets an invocation run part of a pipeline
            while its siblings keep their recorded state. A registry holding entries outside the universe means the
            pipeline's own definition has changed since the tracker was written, so those entries alone are
            discarded and reported through a warning, and every in-universe job keeps its recorded state.

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
            ValueError: If the requested job list is empty, or if any requested job is not part of the resolved
                universe.
        """
        # An empty request is malformed rather than meaningful. Combined with the default universe it would resolve
        # the universe to the empty set, which classifies every tracked job as foreign and discards the whole registry.
        if not jobs:
            message = (
                f"Unable to align the processing tracker at '{self.file_path}' with the requested jobs. The 'jobs' "
                f"argument must name at least one job, but an empty list was provided."
            )
            console.error(message=message, error=ValueError)

        resolved_universe = jobs if universe is None else universe
        universe_ids = {
            self.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in resolved_universe
        }
        requested = [
            (self.generate_job_id(job_name=job_name, specifier=specifier), job_name, specifier)
            for job_name, specifier in jobs
        ]

        out_of_universe = sorted(
            f"{job_name} ({specifier})" if specifier else job_name
            for job_id, job_name, specifier in requested
            if job_id not in universe_ids
        )
        if out_of_universe:
            message = (
                f"Unable to align the processing tracker at '{self.file_path}' with the requested jobs. Every "
                f"requested job must be part of the declared job universe, but the following are absent from it: "
                f"{', '.join(out_of_universe)}."
            )
            console.error(message=message, error=ValueError)

        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            foreign_ids = sorted(set(self.jobs) - universe_ids)
            if foreign_ids:
                message = (
                    f"The processing tracker at '{self.file_path}' contains {len(foreign_ids)} job entries that are "
                    f"not part of the current job universe. Discarding them and preserving every in-universe job. "
                    f"Discarded job IDs: {foreign_ids}."
                )
                with console.temporarily_enabled():
                    console.echo(message=message, level=LogLevel.WARNING)
                for foreign_id in foreign_ids:
                    del self.jobs[foreign_id]

            for job_id, job_name, specifier in requested:
                if job_id not in self.jobs:
                    self.jobs[job_id] = JobState(job_name=job_name, specifier=specifier)

            self._save_state()
            return [job_id for job_id, _, _ in requested]

    def resolve_job(self, job_id: str, universe: list[tuple[str, str]]) -> tuple[str, str]:
        """Resolves the job that a hexadecimal identifier names within the pipeline's declared job universe.

        Notes:
            An invocation handed the identifier of the single job it is to run has to recover that job's name and
            specifier before it is able to execute it. Resolving against the universe rather than against the
            tracker's own registry lets the invocation reject an unknown identifier before the tracker holds any
            entry for it, which keeps a mistyped identifier from registering a job the pipeline cannot produce.

        Args:
            job_id: The unique hexadecimal identifier of the job to resolve.
            universe: The (job_name, specifier) tuples enumerating every job the pipeline could produce for its
                current definition.

        Returns:
            The name and the specifier of the job the identifier names.

        Raises:
            ValueError: If the declared universe is empty, or if the identifier names no job within it.
        """
        if not universe:
            message = (
                f"Unable to resolve the job with ID '{job_id}' against the job universe of the processing tracker at "
                f"'{self.file_path}'. The 'universe' argument must name at least one job, but an empty list was "
                f"provided."
            )
            console.error(message=message, error=ValueError)

        candidates = {
            self.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in universe
        }

        if job_id not in candidates:
            message = (
                f"Unable to resolve the job with ID '{job_id}' against the job universe of the processing tracker at "
                f"'{self.file_path}'. The identifier must name a job the pipeline could produce, but the universe "
                f"holds only the jobs with IDs: {', '.join(sorted(candidates))}."
            )
            console.error(message=message, error=ValueError)

        return candidates[job_id]

    def snapshot(self) -> dict[str, JobState]:
        """Returns a point-in-time copy of the tracker's complete job registry.

        Notes:
            Reads the whole registry under a single lock acquisition, so the returned states are consistent with
            each other.

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

        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            # Copies each state so the caller cannot mutate the instance's registry. Every JobState field is an
            # immutable scalar, so a per-entry replace() is a complete copy and is cheaper than a deep copy.
            return {job_id: replace(job_state) for job_id, job_state in self.jobs.items()}

    def summarize(self) -> dict[str, Any]:
        """Returns the tracker's job registry as per-job details, aggregate job counts, and a progress label.

        Notes:
            Reports every field ``JobState`` carries, so a consumer serializing the returned details preserves the
            registry rather than a projection of it. A job that recorded no failure reason omits the 'error_message'
            key instead of carrying it as an empty value.

        Returns:
            A dictionary carrying the per-job details under 'jobs', the tracked job total alongside the count of the
            jobs holding each status under 'summary', and the label those counts resolve to under 'status'.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        registry = self.snapshot()

        job_details: list[dict[str, Any]] = [
            {
                "job_id": job_id,
                "job_name": job_state.job_name,
                "specifier": job_state.specifier,
                "status": job_state.status.name,
                "executor_id": job_state.executor_id,
                "started_at": job_state.started_at,
                "completed_at": job_state.completed_at,
                **({} if job_state.error_message is None else {"error_message": job_state.error_message}),
            }
            for job_id, job_state in registry.items()
        ]

        counts: dict[str, int] = {
            "total": len(registry),
            "succeeded": sum(1 for state in registry.values() if state.status == ProcessingStatus.SUCCEEDED),
            "failed": sum(1 for state in registry.values() if state.status == ProcessingStatus.FAILED),
            "running": sum(1 for state in registry.values() if state.status == ProcessingStatus.RUNNING),
            "scheduled": sum(1 for state in registry.values() if state.status == ProcessingStatus.SCHEDULED),
        }

        return {"jobs": job_details, "summary": counts, "status": self.resolve_status(summary=counts)}

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
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            return {
                job_id: (job_state.job_name, job_state.specifier)
                for job_id, job_state in self.jobs.items()
                if (job_name is None or job_name in job_state.job_name)
                and (specifier is None or specifier in job_state.specifier)
            }

    @contextmanager
    def run_job(self, job_id: str, executor_id: str | None = None) -> Iterator[None]:
        """Runs a single tracked job, recording its start, its completion, and its failure on the tracker.

        Notes:
            Owns the job's state transitions and leaves the work itself to the wrapped block. The guard spans both
            the block and the completion call, so an ``Exception`` raised by either marks the job failed, records the
            exception's message as the failure reason, and re-raises the exception unchanged.

            A ``BaseException`` such as ``KeyboardInterrupt`` propagates with the job left running, since an
            interrupted job did not fail on its own terms and its executor is the authority on what became of it.

        Args:
            job_id: The unique identifier of the job to run.
            executor_id: An optional explicit identifier for the executor running the job. When None (default), the
                identifier is resolved automatically from the runtime environment.

        Yields:
            None. The tracker holds the job in its running state for the duration of the block.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        self.start_job(job_id=job_id, executor_id=executor_id)
        try:
            yield
            self.complete_job(job_id=job_id)
        except Exception as exception:
            self.fail_job(job_id=job_id, error_message=str(exception))
            raise

    def start_job(self, job_id: str, executor_id: str | None = None) -> None:
        """Marks the target job as running and records the identifier of the executor running it.

        Clears the error message and completion timestamp recorded by any previous attempt at the job.

        Args:
            job_id: The unique identifier of the job to mark as started.
            executor_id: An optional explicit identifier for the executor running the job. When None (default), the
                identifier is resolved automatically from the runtime environment, preferring a recognized job
                scheduler's job ID and falling back to the process ID, each tagged with its scheme.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"Unable to mark the job with ID '{job_id}' as running using the processing tracker at "
                    f"'{self.file_path}'. The requested job must be tracked by the instance, but the instance is "
                    f"not configured to track it. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.RUNNING
            job_info.error_message = None
            job_info.completed_at = None
            job_info.executor_id = executor_id if executor_id is not None else self._resolve_executor_id()
            job_info.started_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND),
            )

            self._save_state()

    def complete_job(self, job_id: str) -> None:
        """Marks a target job as successfully completed.

        Clears the error message recorded by any previous attempt at the job.

        Args:
            job_id: The unique identifier of the job to mark as complete.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"Unable to mark the job with ID '{job_id}' as complete using the processing tracker at "
                    f"'{self.file_path}'. The requested job must be tracked by the instance, but the instance is "
                    f"not configured to track it. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.SUCCEEDED
            job_info.error_message = None
            job_info.completed_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND),
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
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"Unable to mark the job with ID '{job_id}' as failed using the processing tracker at "
                    f"'{self.file_path}'. The requested job must be tracked by the instance, but the instance is "
                    f"not configured to track it. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            job_info = self.jobs[job_id]
            job_info.status = ProcessingStatus.FAILED
            job_info.error_message = error_message
            job_info.completed_at = int(
                get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND),
            )

            self._save_state()

    def get_job_status(self, job_id: str) -> ProcessingStatus:
        """Queries the current runtime status of the target job.

        Args:
            job_id: The unique identifier of the job for which to query the runtime status.

        Returns:
            The status the tracker currently records for the target job.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"Unable to query the status of the job with ID '{job_id}' using the processing tracker at "
                    f"'{self.file_path}'. The requested job must be tracked by the instance, but the instance is "
                    f"not configured to track it. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            return self.jobs[job_id].status

    def reset(self) -> None:
        """Resets the tracker file to the default state.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            self.jobs.clear()
            self._save_state()

    @property
    def complete(self) -> bool:
        """Returns True when the tracked pipeline has jobs and all of them have been marked as succeeded.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()
            if not self.jobs:
                return False
            return all(job.status == ProcessingStatus.SUCCEEDED for job in self.jobs.values())

    @property
    def encountered_error(self) -> bool:
        """Returns True when any of the tracked pipeline's jobs has been marked as failed.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()
            return any(job.status == ProcessingStatus.FAILED for job in self.jobs.values())

    def get_jobs_by_status(self, status: ProcessingStatus | str) -> list[str]:
        """Returns all job IDs that have the specified status.

        Args:
            status: The status to match, given as a ``ProcessingStatus`` member or its member name string.

        Returns:
            The identifiers of every tracked job currently holding the requested status.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            KeyError: If ``status`` is a string that does not name a valid ProcessingStatus member.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()
            target_status = ProcessingStatus[status] if isinstance(status, str) else status
            return [job_id for job_id, job_state in self.jobs.items() if job_state.status == target_status]

    def get_summary(self) -> dict[ProcessingStatus, int]:
        """Returns a summary of job counts by status.

        Returns:
            The number of tracked jobs currently holding each status, with every status present even when its
            count is zero.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()
            summary: dict[ProcessingStatus, int] = dict.fromkeys(ProcessingStatus, 0)
            for job_state in self.jobs.values():
                summary[job_state.status] += 1
            return summary

    def get_job_info(self, job_id: str) -> JobState:
        """Returns the full ``JobState`` object for the specified job.

        Notes:
            The returned state is a copy, so mutating it does not affect the tracker. Use the dedicated mutators to
            change job state.

        Args:
            job_id: The unique identifier of the job to query.

        Returns:
            A copy of the ``JobState`` object containing all metadata for the job.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
            ValueError: If the specified job ID is not found in the managed tracker file.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            if job_id not in self.jobs:
                message = (
                    f"Unable to query the state of the job with ID '{job_id}' using the processing tracker at "
                    f"'{self.file_path}'. The requested job must be tracked by the instance, but the instance is "
                    f"not configured to track it. The instance is currently configured to track jobs with IDs: "
                    f"{', '.join(self.jobs.keys())}."
                )
                console.error(message=message, error=ValueError)

            # Every JobState field is an immutable scalar, so replace() is a complete copy.
            return replace(self.jobs[job_id])

    def reset_jobs(self, job_ids: list[str]) -> list[str]:
        """Resets the specified jobs back to SCHEDULED status, leaving every other job's state untouched.

        Clears the ``error_message``, ``started_at``, ``completed_at``, and ``executor_id`` fields of each
        targeted job.

        Notes:
            Every job outside ``job_ids`` keeps its recorded status, executor, and timestamps.

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

        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
            self._load_state()

            unknown_ids = sorted(targeted - set(self.jobs))
            if unknown_ids:
                message = (
                    f"Unable to reset the jobs using the processing tracker at '{self.file_path}'. Every requested "
                    f"job must be tracked by the instance, but the instance is not configured to track the job(s) "
                    f"with ID(s): {', '.join(unknown_ids)}. The instance is currently configured to track jobs with "
                    f"IDs: {', '.join(self.jobs.keys())}."
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

        Clears the ``error_message``, ``started_at``, ``completed_at``, and ``executor_id`` fields for each
        failed job.

        Returns:
            A list of job IDs that were reset for retry.

        Raises:
            TimeoutError: If the .LOCK file for the tracker .YAML file cannot be acquired within the timeout period.
        """
        lock = FileLock(lock_file=self.lock_path)
        with lock.acquire(timeout=_LOCK_ACQUISITION_TIMEOUT):
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

    @staticmethod
    def _resolve_executor_id() -> str:
        """Resolves the identifier of the executor running a job from the runtime environment.

        Consults the recognized job schedulers in ``_SCHEDULER_EXECUTOR_SOURCES`` in priority order and returns the
        first scheduler job ID found, tagged with its scheme as ``"<scheme>:<id>"``. The tag lets a stale tracker
        entry be correlated with the scheduler's own record of the job. Falls back to ``"pid:<process id>"`` when the
        process runs under no recognized scheduler, so a locally executed job still records a meaningful executor
        identifier.

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
        """Reads the processing pipeline's runtime state from the cached .YAML file, creating the file with the
        instance's current state when it does not yet exist.
        """
        if self.file_path.exists():
            loaded = ProcessingTracker.from_yaml(file_path=self.file_path)
            self.jobs = loaded.jobs
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Caches the current processing state stored inside the instance's attributes as a .YAML file."""
        # Both path fields carry the exclusion marker, so the emitted document holds the job registry alone.
        self.to_yaml(file_path=self.file_path)
