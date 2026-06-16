"""Provides assets for moving data between filesystem destinations and removing data from the host machine."""

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console, resolve_worker_count, ensure_directory_exists

from .checksum_tools import calculate_directory_checksum

_MAXIMUM_DELETION_ATTEMPTS: int = 5
"""The maximum number of times directory deletion is retried before giving up."""

_DELETION_RETRY_DELAY_MS: int = 500
"""The delay in milliseconds between failed directory-deletion attempts."""


def delete_directory(directory_path: Path) -> None:
    """Deletes the target directory and all its subdirectories, unlinking the files within each directory in parallel.

    Args:
        directory_path: The path to the directory to delete.
    """
    if not directory_path.exists():
        return

    files = [path for path in directory_path.iterdir() if path.is_file()]
    subdirectories = [path for path in directory_path.iterdir() if path.is_dir()]

    with ThreadPoolExecutor() as executor:
        list(executor.map(Path.unlink, files))  # Forces completion of all tasks

    for subdirectory in subdirectories:
        delete_directory(directory_path=subdirectory)

    # Removes the now-empty root directory. Retries with a short delay between attempts, because Windows can be slow
    # to release file handles.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    for _ in range(_MAXIMUM_DELETION_ATTEMPTS):
        try:
            directory_path.rmdir()
            break
        except Exception:  # noqa: BLE001  # pragma: no cover
            delay_timer.delay(block=False, delay=_DELETION_RETRY_DELAY_MS, allow_sleep=True)
            continue


def _transfer_file(source_file: Path, source_directory: Path, destination_directory: Path) -> None:
    """Copies the input file from the source directory to the destination directory while preserving the file metadata.

    This worker function is used by the transfer_directory() function to move multiple files in parallel.

    Notes:
        If the file is found under a hierarchy of subdirectories inside the input source_directory, that hierarchy will
        be preserved in the destination directory.

    Args:
        source_file: The file to be copied.
        source_directory: The root directory where the file is located.
        destination_directory: The destination directory where to move the file.
    """
    relative = source_file.relative_to(source_directory)
    destination_file = destination_directory / relative
    shutil.copy2(src=source_file, dst=destination_file)


def _collect_source_items(source: Path) -> tuple[list[Path], list[Path]]:
    """Discovers the contents of the source directory and separates them into subdirectories and files.

    Notes:
        Both lists are sorted by path depth so that parent directories precede their children and the file copy
        order is deterministic. Any item that is not a directory is treated as a file.

    Args:
        source: The root directory whose contents are discovered.

    Returns:
        The subdirectories and files found anywhere under the source directory, each sorted by path depth.
    """
    all_items = sorted(source.rglob("*"), key=lambda path: len(path.relative_to(source).parts))
    subdirectories = [item for item in all_items if item.is_dir()]
    files = [item for item in all_items if not item.is_dir()]
    return subdirectories, files


def _plan_destination_directories(source: Path, destination: Path, subdirectories: list[Path]) -> list[Path]:
    """Maps each source subdirectory to its corresponding path inside the destination directory.

    Args:
        source: The root source directory the subdirectories are relative to.
        destination: The root destination directory the hierarchy is recreated under.
        subdirectories: The source subdirectories to map to destination paths.

    Returns:
        The destination directory paths that recreate the source subdirectory hierarchy.
    """
    return [destination / subdirectory.relative_to(source) for subdirectory in subdirectories]


def transfer_directory(
    source: Path,
    destination: Path,
    num_threads: int = 1,
    *,
    verify_integrity: bool = False,
    remove_source: bool = False,
    progress: bool = False,
) -> None:
    """Copies the contents of the input source directory to the destination directory while preserving the underlying
    directory hierarchy.

    Notes:
        This function recreates the moved directory hierarchy on the destination if the hierarchy does not exist. This
        is done before copying the files.

        The function performs a multithreaded copy operation when 'num_threads' is greater than 1 and a sequential
        copy otherwise. By default, it does not remove the source data after the copy is complete.

        If the function is configured to verify the transferred data's integrity, it generates an xxHash3-128 checksum
        of the data before and after the transfer and compares the two checksums to detect data corruption.

    Args:
        source: The path to the directory to be transferred.
        destination: The path to the destination directory where to move the contents of the source directory.
        num_threads: The number of threads to use for the parallel file transfer. Setting this value below 1 instructs
            the function to use all available CPU cores minus a small number reserved for the host system.
        verify_integrity: Determines whether to perform integrity verification for the transferred files.
        remove_source: Determines whether to remove the source directory after the transfer is complete and
            (optionally) verified.
        progress: Determines whether to track the transfer progress using a progress bar.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        RuntimeError: If the transferred files do not pass the xxHash3-128 checksum integrity verification.
    """
    if not source.exists():
        message = f"Unable to transfer the source directory {source}, as it does not exist."
        console.error(message=message, error=FileNotFoundError)

    # If the number of threads is less than 1, uses all available CPU cores minus a small number reserved for the
    # host system.
    if num_threads < 1:
        num_threads = resolve_worker_count()

    # If transfer integrity verification is enabled, but the source directory does not contain the 'ax_checksum.txt'
    # file, checksums the directory before the transfer operation.
    if verify_integrity and not source.joinpath("ax_checksum.txt").exists():
        calculate_directory_checksum(directory=source, progress=False, save_checksum=True)

    ensure_directory_exists(path=destination)

    # Discovers the source directory contents and recreates its subdirectory hierarchy inside the destination before
    # copying any files.
    subdirectories, file_list = _collect_source_items(source=source)
    for destination_directory_path in _plan_destination_directories(
        source=source, destination=destination, subdirectories=subdirectories
    ):
        destination_directory_path.mkdir(parents=True, exist_ok=True)

    # Copies the data to the destination. For parallel workflows, the method uses the ThreadPoolExecutor to move
    # multiple files at the same time. Since I/O operations do not hold GIL, we do not need to parallelize with
    # Processes here.
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    _transfer_file,
                    source_file=file,
                    source_directory=source,
                    destination_directory=destination,
                ): file
                for file in file_list
            }
            if progress:  # pragma: no cover
                with console.progress(
                    total=len(file_list),
                    description=f"Transferring files to {destination.name}",
                    unit="file",
                ) as progress_bar:
                    for future in as_completed(futures):
                        future.result()  # Propagates any exceptions from the file transfer.
                        progress_bar.update(1)
            else:
                for future in as_completed(futures):
                    future.result()
    elif progress:
        for file in console.track(file_list, description=f"Transferring files to {destination.name}", unit="file"):
            _transfer_file(source_file=file, source_directory=source, destination_directory=destination)
    else:
        for file in file_list:
            _transfer_file(source_file=file, source_directory=source, destination_directory=destination)

    # Verifies the integrity of the transferred directory by rerunning xxHash3-128 calculation.
    if verify_integrity:
        destination_checksum = calculate_directory_checksum(directory=destination, progress=False, save_checksum=False)
        with source.joinpath("ax_checksum.txt").open("r") as local_checksum:
            if destination_checksum != local_checksum.readline().strip():
                message = (
                    f"Checksum mismatch detected when transferring {Path(*source.parts[-6:])} to "
                    f"{Path(*destination.parts[-6:])}! The data was likely corrupted in transmission."
                )
                console.error(message=message, error=RuntimeError)

    # If necessary, removes the transferred directory from the original location.
    if remove_source:
        message = (
            f"Removing the now-redundant source directory {source} and all of its contents following the successful "
            f"transfer..."
        )
        console.echo(message=message)
        delete_directory(directory_path=source)
