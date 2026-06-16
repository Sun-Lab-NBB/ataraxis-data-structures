from pathlib import Path

from .checksum_tools import calculate_directory_checksum as calculate_directory_checksum

_MAXIMUM_DELETION_ATTEMPTS: int
_DELETION_RETRY_DELAY_MS: int

def delete_directory(directory_path: Path) -> None: ...
def _transfer_file(source_file: Path, source_directory: Path, destination_directory: Path) -> None: ...
def _collect_source_items(source: Path) -> tuple[list[Path], list[Path]]: ...
def _plan_destination_directories(source: Path, destination: Path, subdirectories: list[Path]) -> list[Path]: ...
def transfer_directory(
    source: Path,
    destination: Path,
    num_threads: int = 1,
    *,
    verify_integrity: bool = False,
    remove_source: bool = False,
    progress: bool = False,
) -> None: ...
