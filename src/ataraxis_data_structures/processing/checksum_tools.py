"""Provides assets for computing data integrity checksums."""

from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import xxhash
from ataraxis_base_utilities import console, resolve_worker_count


def _calculate_file_checksum(base_directory: Path, file_path: Path) -> tuple[str, bytes]:  # pragma: no cover
    """Calculates the xxHash3-128 checksum for the target file and its path relative to the base directory.

    Args:
        base_directory: The path to the directory being processed by the 'calculate_directory_checksum' function.
        file_path: The path to the target file relative to the base directory.

    Returns:
        A tuple with two elements. The first element is the path to the file relative to the base directory. The second
        element is the xxHash3-128 checksum that reflects the file's path and data.
    """
    # Initializes the hashsum object.
    checksum = xxhash.xxh3_128()

    # Encodes the relative path and appends it to the checksum. This ensures that the hashsum reflects both the state
    # of individual files and the layout of the overall encoded directory structure.
    relative_path = str(file_path.relative_to(base_directory))
    checksum.update(relative_path.encode())

    # Extends the checksum to reflect the file data state. Uses 8 MB chunks to avoid excessive RAM hogging at the cost
    # of slightly reduced throughput.
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            checksum.update(chunk)

    # Returns both path and file checksum. Although the relative path information is already encoded in the hashsum, the
    # relative path information is re-encoded at the directory level to protect against future changes to the per-file
    # hashsum calculation logic. It is extra work, but it improves the overall checksum security.
    return relative_path, checksum.digest()


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
            function uses all available CPU cores.
        progress: Determines whether to track the checksum calculation progress using a progress bar.
        save_checksum: Determines whether the checksum should be saved (written to) a .txt file.
        excluded_files: The set of filenames to exclude from the checksum calculation. If set to None, defaults to
            ``{"ax_checksum.txt"}``.

    Returns:
        The xxHash3-128 checksum for the input directory as a hexadecimal string.
    """
    # Resolves excluded files default.
    if excluded_files is None:
        excluded_files = {"ax_checksum.txt"}

    # Determines the number of parallel processes to use.
    if num_processes is None:
        num_processes = resolve_worker_count()

    # Determines the path to each file inside the input directory structure and sorts them for consistency
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and f"{path.stem}{path.suffix}" not in excluded_files  # Excludes service files
    )

    # Pre-creates the directory checksum
    checksum = xxhash.xxh3_128()

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Creates the partial function with fixed base_directory (the first argument of _calculate_file_hash())
        process_file = partial(_calculate_file_checksum, directory)

        # Submits all tasks to be executed in parallel
        # noinspection PyTypeChecker
        future_to_path = {executor.submit(process_file, file): file for file in files}

        # Collects results as they complete
        results = []
        if progress:
            with console.progress(
                total=len(files), description=f"Calculating checksum for {directory.name}", unit="file"
            ) as pbar:
                for future in as_completed(future_to_path):
                    results.append(future.result())
                    pbar.update(1)
        else:
            # For batch mode, uses a direct list comprehension with as_completed. This avoids the overhead of progress
            # tracking while maintaining parallel processing, avoiding terminal clutter in batched contexts.
            results = [future.result() for future in as_completed(future_to_path)]

        # Sorts results for consistency and combines them into the final checksum
        for file_path, file_checksum in sorted(results):
            checksum.update(file_path.encode())
            checksum.update(file_checksum)

    checksum_hexstring = checksum.hexdigest()

    # Writes the hash to ax_checksum.txt in the root directory
    if save_checksum:
        checksum_path = directory.joinpath("ax_checksum.txt")
        with checksum_path.open("w") as f:
            f.write(checksum_hexstring)

    return checksum_hexstring
