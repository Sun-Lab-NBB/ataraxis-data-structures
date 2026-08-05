"""Provides assets for computing data integrity checksums."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import xxhash
from ataraxis_base_utilities import console, resolve_worker_count

from .parallel_tools import limit_worker_threads

if TYPE_CHECKING:
    from pathlib import Path
    from multiprocessing.context import SpawnContext


_MULTIPROCESSING_CONTEXT: SpawnContext = get_context("spawn")
"""The spawn-based multiprocessing context used to create the process pool that calculates file checksums, ensuring
identical cross-platform behavior on all supported platforms."""

CHECKSUM_FILENAME: str = "ax_checksum.txt"
"""The name of the file a directory checksum is written to, at the top level of the checksummed directory.

Notes:
    The name is shared with the transfer utilities, which both exclude it from the digest and treat it as an expected
    destination entry, so both sides have to agree on the spelling.
"""

_CHECKSUM_CHUNK_SIZE: int = 1024 * 1024 * 8
"""The size, in bytes, of the buffer each worker reads file data into. Bounds the resident memory one worker needs to
checksum a file of any size."""


def calculate_directory_checksum(
    directory: Path,
    num_processes: int | None = None,
    *,
    progress: bool = False,
    save_checksum: bool = True,
    excluded_files: set[str] | None = None,
) -> str:
    """Calculates the xxHash3-128 checksum for the input directory.

    Notes:
        The function can be configured to write the generated checksum as a hexadecimal string to the ax_checksum.txt
        file stored at the highest level of the input directory.

        The xxHash3 checksum is not suitable for security purposes and is only used to ensure data integrity.

        The returned checksum accounts for both the contents of each file and the layout of the input directory
        structure.

    Args:
        directory: The path to the directory for which to generate the checksum.
        num_processes: The number of processes to use for parallelizing checksum calculation. If set to None, the
            function uses all available CPU cores minus 2 reserved cores (via ``resolve_worker_count``).
        progress: Determines whether to track the checksum calculation progress using a progress bar.
        save_checksum: Determines whether to write the checksum to the ax_checksum.txt file at the top level of the
            input directory.
        excluded_files: The set of filenames to exclude from the checksum calculation. If set to None, defaults to
            ``{"ax_checksum.txt"}``.

    Returns:
        The xxHash3-128 checksum for the input directory as a hexadecimal string.
    """
    if excluded_files is None:
        excluded_files = {CHECKSUM_FILENAME}

    if num_processes is None:
        num_processes = resolve_worker_count()

    files = _discover_checksum_files(directory=directory, excluded_files=excluded_files)

    checksum = xxhash.xxh3_128()

    with (
        limit_worker_threads(),
        ProcessPoolExecutor(max_workers=num_processes, mp_context=_MULTIPROCESSING_CONTEXT) as executor,
    ):
        # Binds base_directory so each submitted task only needs to supply the per-file path.
        process_file = partial(_calculate_file_checksum, base_directory=directory)

        future_to_path = {executor.submit(process_file, file_path=file): file for file in files}

        results = []
        if progress:
            with console.progress(
                total=len(files),
                description=f"Calculating checksum for {directory.name}",
                unit="file",
            ) as progress_bar:
                for future in as_completed(future_to_path):
                    results.append(future.result())
                    progress_bar.update(n=1)
        else:
            # Skips progress tracking in batch mode to avoid its overhead and to keep batched contexts free of
            # terminal clutter.
            results = [future.result() for future in as_completed(future_to_path)]

        # Sorts results for consistency, so that the combined directory checksum does not depend on completion order.
        for file_path, file_checksum in sorted(results):
            checksum.update(file_path.encode())
            checksum.update(file_checksum)

    checksum_hexstring = checksum.hexdigest()

    if save_checksum:
        _write_checksum_file(directory=directory, checksum=checksum_hexstring)

    return checksum_hexstring


def _calculate_file_checksum(base_directory: Path, file_path: Path) -> tuple[str, bytes]:
    """Calculates the xxHash3-128 checksum for the target file and its path relative to the base directory.

    Args:
        base_directory: The path to the directory that contains the target file and anchors its relative path.
        file_path: The full path to the target file located inside the base directory.

    Returns:
        The first element is the file path relative to the base directory. The second is the xxHash3-128 checksum
        reflecting the file's path and data.
    """
    checksum = xxhash.xxh3_128()

    # Encodes the relative path and appends it to the checksum. This ensures that the hashsum reflects both the state
    # of individual files and the layout of the overall encoded directory structure.
    relative_path = str(file_path.relative_to(base_directory))
    checksum.update(relative_path.encode())

    # Extends the checksum to reflect the file data state. Uses 8 MB chunks to avoid excessive RAM hogging at the cost
    # of slightly reduced throughput. Reads into one reusable buffer, so a large file costs a single allocation
    # instead of one per chunk.
    chunk_buffer = bytearray(_CHECKSUM_CHUNK_SIZE)
    chunk_view = memoryview(chunk_buffer)
    with file_path.open("rb") as file:
        while (read_byte_count := file.readinto(chunk_buffer)) > 0:
            checksum.update(chunk_view[:read_byte_count])

    # Returns both path and file checksum. Although the relative path information is already encoded in the hashsum, the
    # relative path information is re-encoded at the directory level to protect against future changes to the per-file
    # hashsum calculation logic. It is extra work, but it improves the overall checksum security.
    return relative_path, checksum.digest()


def _discover_checksum_files(directory: Path, excluded_files: set[str]) -> list[Path]:
    """Discovers the files to include in a directory checksum, sorted for order-independent hashing.

    Args:
        directory: The directory whose files are discovered.
        excluded_files: The set of filenames to omit from the checksum.

    Returns:
        The files found anywhere under the directory, excluding the omitted filenames, sorted by path.
    """
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and f"{path.stem}{path.suffix}" not in excluded_files
    )


def _write_checksum_file(directory: Path, checksum: str) -> None:
    """Writes the directory checksum as a hexadecimal string to the ax_checksum.txt file at the directory's top level.

    Args:
        directory: The directory whose top level receives the ax_checksum.txt file.
        checksum: The hexadecimal checksum string to write.
    """
    checksum_path = directory.joinpath(CHECKSUM_FILENAME)
    with checksum_path.open("w") as file:
        file.write(checksum)
