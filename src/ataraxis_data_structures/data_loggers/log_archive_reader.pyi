from pathlib import Path
from functools import cached_property as cached_property
from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray as NDArray

@dataclass(frozen=True, slots=True)
class LogMessage:
    timestamp_us: np.uint64
    payload: NDArray[np.uint8]

class LogArchiveReader:
    _TIMESTAMP_BYTE_SIZE: int
    _PARALLEL_PROCESSING_THRESHOLD: int
    _ONSET_KEY_SUFFIX: str
    _archive_path: Path
    _onset_us: np.uint64 | None
    _message_keys: list[str] | None
    def __init__(self, archive_path: Path, onset_us: np.uint64 | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @cached_property
    def onset_timestamp_us(self) -> np.uint64: ...
    def _get_message_keys(self) -> list[str]: ...
    @property
    def message_count(self) -> int: ...
    def get_batches(self, workers: int = -1, batch_multiplier: int = 4) -> list[list[str]]: ...
    def iter_messages(self, keys: list[str] | None = None) -> Iterator[LogMessage]: ...
    def read_all_messages(self) -> tuple[NDArray[np.uint64], list[NDArray[np.uint8]]]: ...
