"""Provides classes and structures for storing, manipulating, and sharing data between Python processes.

See the `API documentation <https://ataraxis-data-structures-api-docs.netlify.app/>`_ for the description of
available classes and structures. See the `source code repository
<https://github.com/Sun-Lab-NBB/ataraxis-data-structures>`_ for more details.

Authors: Ivan Kondratyev (Inkaros)
"""

from .processing import (
    delete_directory,
    interpolate_data,
    transfer_directory,
    limit_worker_threads,
    resolve_unique_roots,
    discover_marker_files,
    discover_marker_roots,
    initialize_worker_threads,
    calculate_directory_checksum,
)
from .data_loggers import (
    LOG_ARCHIVE_SUFFIX,
    LOG_DIRECTORY_SUFFIX,
    PARALLEL_PROCESSING_THRESHOLD,
    DataLogger,
    LogMessage,
    LogPackage,
    LogArchiveReader,
    find_log_archive,
    assemble_log_archives,
    discover_log_archives,
    read_archive_message_count,
)
from .shared_memory import SharedMemoryArray
from .data_structures import (
    YAML_EXCLUDE_METADATA,
    JobState,
    YamlConfig,
    TrackerStatus,
    ProcessingStatus,
    ProcessingTracker,
)

__all__ = [
    "LOG_ARCHIVE_SUFFIX",
    "LOG_DIRECTORY_SUFFIX",
    "PARALLEL_PROCESSING_THRESHOLD",
    "YAML_EXCLUDE_METADATA",
    "DataLogger",
    "JobState",
    "LogArchiveReader",
    "LogMessage",
    "LogPackage",
    "ProcessingStatus",
    "ProcessingTracker",
    "SharedMemoryArray",
    "TrackerStatus",
    "YamlConfig",
    "assemble_log_archives",
    "calculate_directory_checksum",
    "delete_directory",
    "discover_log_archives",
    "discover_marker_files",
    "discover_marker_roots",
    "find_log_archive",
    "initialize_worker_threads",
    "interpolate_data",
    "limit_worker_threads",
    "read_archive_message_count",
    "resolve_unique_roots",
    "transfer_directory",
]
