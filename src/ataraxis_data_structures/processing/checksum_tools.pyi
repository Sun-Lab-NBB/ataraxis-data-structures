from pathlib import Path

def _calculate_file_checksum(base_directory: Path, file_path: Path) -> tuple[str, bytes]: ...
def calculate_directory_checksum(
    directory: Path,
    num_processes: int | None = None,
    *,
    progress: bool = False,
    save_checksum: bool = True,
    excluded_files: set[str] | None = None,
) -> str: ...
