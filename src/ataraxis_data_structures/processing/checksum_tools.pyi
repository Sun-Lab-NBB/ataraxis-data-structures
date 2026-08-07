from pathlib import Path
from multiprocessing.context import SpawnContext

from .parallel_tools import limit_worker_threads as limit_worker_threads
from .filesystem_tools import walk_files as walk_files

_MULTIPROCESSING_CONTEXT: SpawnContext
CHECKSUM_FILENAME: str
_CHECKSUM_CHUNK_SIZE: int

def calculate_directory_checksum(
    directory: Path,
    num_processes: int | None = None,
    *,
    progress: bool = False,
    save_checksum: bool = True,
    excluded_files: set[str] | None = None,
) -> str: ...
def _discover_checksum_files(directory: Path, excluded_files: set[str]) -> list[Path]: ...
def _calculate_file_checksum(base_directory: Path, file_path: Path) -> tuple[str, bytes]: ...
def _write_checksum_file(directory: Path, checksum: str) -> None: ...
