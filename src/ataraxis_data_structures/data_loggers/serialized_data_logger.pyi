from typing import Any
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray

from ..shared_memory import SharedMemoryArray as SharedMemoryArray

def _load_numpy_files(
    file_paths: tuple[Path, ...], *, memory_map: bool = False
) -> tuple[tuple[str, ...], tuple[NDArray[Any], ...]]: ...
def _load_numpy_archive(file_path: Path) -> dict[str, NDArray[Any]]: ...
def _assemble_archive(
    output_directory: Path, source_id: int, source_data: dict[str, NDArray[Any]]
) -> tuple[int, Path]: ...
def _compare_arrays(source_id: int, stem: str, original_array: NDArray[Any], archived_array: NDArray[Any]) -> None: ...
def assemble_log_archives(
    log_directory: Path,
    max_workers: int | None = None,
    *,
    remove_sources: bool = True,
    memory_mapping: bool = True,
    verbose: bool = False,
    verify_integrity: bool = False,
) -> None: ...
@dataclass(frozen=True)
class LogPackage:
    source_id: np.uint8
    acquisition_time: np.uint64
    serialized_data: NDArray[np.uint8]
    @property
    def data(self) -> tuple[str, NDArray[np.uint8]]: ...

class DataLogger:
    _started: bool
    _mp_manager: SyncManager
    _thread_count: int
    _poll_interval: int
    _name: Incomplete
    _output_directory: Path
    _input_queue: MPQueue
    _terminator_array: SharedMemoryArray | None
    _logger_process: Process | None
    _watchdog_thread: Thread | None
    def __init__(
        self, output_directory: Path, instance_name: str, thread_count: int = 5, poll_interval: int = 5
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __del__(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def _watchdog(self) -> None: ...
    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None: ...
    @staticmethod
    def _log_cycle(
        input_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        output_directory: Path,
        thread_count: int,
        poll_interval: int,
    ) -> None: ...
    @property
    def input_queue(self) -> MPQueue: ...
    @property
    def name(self) -> str: ...
    @property
    def alive(self) -> bool: ...
    @property
    def output_directory(self) -> Path: ...
