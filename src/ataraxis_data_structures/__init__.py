"""Provides classes and structures for storing, manipulating, and sharing data between Python processes.

See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-data-structures>`_ for more details.
API documentation: `ataraxis-data-structures-api-docs <https://ataraxis-data-structures-api-docs.netlify.app/>`_

Authors: Ivan Kondratyev (Inkaros)
"""

from .processing import delete_directory, interpolate_data, transfer_directory, calculate_directory_checksum
from .data_loggers import DataLogger, LogPackage, assemble_log_archives
from .shared_memory import SharedMemoryArray
from .data_structures import JobState, YamlConfig, ProcessingStatus, ProcessingTracker

__all__ = [
    "DataLogger",
    "JobState",
    "LogPackage",
    "ProcessingStatus",
    "ProcessingTracker",
    "SharedMemoryArray",
    "YamlConfig",
    "assemble_log_archives",
    "calculate_directory_checksum",
    "delete_directory",
    "interpolate_data",
    "transfer_directory",
]
