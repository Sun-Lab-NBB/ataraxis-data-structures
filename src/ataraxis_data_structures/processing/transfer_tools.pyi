from pathlib import Path

from .checksum_tools import (
    CHECKSUM_FILENAME as CHECKSUM_FILENAME,
    calculate_directory_checksum as calculate_directory_checksum,
)
from .filesystem_tools import (
    walk_files as walk_files,
    walk_directory as walk_directory,
    reports_absent_entry as reports_absent_entry,
)

_MAXIMUM_DELETION_ATTEMPTS: int
_DELETION_RETRY_DELAY_MILLISECONDS: int

def delete_directory(directory_path: Path) -> None: ...
def transfer_directory(
    source: Path,
    destination: Path,
    num_threads: int = 1,
    *,
    verify_integrity: bool = False,
    remove_source: bool = False,
    progress: bool = False,
    reset_dirty_destination: bool = False,
) -> None: ...
def _collect_source_items(source: Path) -> tuple[list[Path], list[Path], list[Path]]: ...
def _classify_entry(path: Path) -> tuple[bool, bool]: ...
def _find_unaccounted_destination_files(source: Path, destination: Path, source_files: list[Path]) -> list[Path]: ...
def _plan_destination_directories(source: Path, destination: Path, subdirectories: list[Path]) -> list[Path]: ...
def _transfer_file(source_file: Path, source_directory: Path, destination_directory: Path) -> None: ...
