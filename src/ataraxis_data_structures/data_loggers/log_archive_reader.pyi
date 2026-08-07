from pathlib import Path
from functools import cached_property as cached_property
from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray as NDArray

from ..processing import discover_marker_files as discover_marker_files
from .serialized_data_logger import LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX

PARALLEL_PROCESSING_THRESHOLD: int
_TIMESTAMP_BYTE_SIZE: int
_ONSET_KEY_SUFFIX: str

@dataclass(frozen=True, slots=True)
class LogMessage:
    timestamp_us: np.uint64
    payload: NDArray[np.uint8]

def find_log_archive(log_directory: Path, source_id: str) -> Path: ...
def discover_log_archives(log_directory: Path) -> dict[str, Path]: ...
def read_archive_message_count(archive_path: Path) -> int: ...

class LogArchiveReader:
    _archive_path: Path
    _onset_us: np.uint64 | None
    _message_keys: list[str] | None
    def __init__(self, archive_path: Path, onset_us: np.uint64 | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @cached_property
    def onset_timestamp_us(self) -> np.uint64: ...
    @property
    def message_count(self) -> int: ...
    def get_batches(self, workers: int = -1, batch_multiplier: int = 4) -> list[list[str]]: ...
    def iter_messages(self, keys: list[str] | None = None) -> Iterator[LogMessage]: ...
    def read_all_messages(self) -> tuple[NDArray[np.uint64], list[NDArray[np.uint8]]]: ...
    def _get_message_keys(self) -> list[str]: ...
