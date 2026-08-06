"""Provides assets for moving data between filesystem destinations and removing data from the host machine."""

import os
import stat
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count, ensure_directory_exists

from .checksum_tools import CHECKSUM_FILENAME, calculate_directory_checksum
from .filesystem_tools import walk_files, walk_directory, reports_absent_entry

_MAXIMUM_DELETION_ATTEMPTS: int = 5
"""The maximum number of times directory removal is attempted before giving up."""

_DELETION_RETRY_DELAY_MILLISECONDS: int = 500
"""The delay in milliseconds between failed directory-deletion attempts."""


def delete_directory(directory_path: Path) -> None:
    """Deletes the target directory and all its subdirectories, unlinking the files within each directory in parallel.

    Notes:
        A symlink is removed as a link, whatever it points at, so the tree behind a symlinked subdirectory is left
        untouched and only entries living inside the target directory are deleted. The rule covers the target itself,
        so a link passed as ``directory_path`` is unlinked in place and the tree it names is left whole. Every entry
        that is not a real directory is unlinked in place, which additionally covers the sockets and FIFOs that a file
        check skips.

        Removal of each emptied directory is attempted up to five times, with a 500 millisecond delay between
        attempts, as some operating systems are slow to release file handles. If every attempt fails, the function
        reports a warning and returns, leaving the directory in place. The path is verifiable with Path.exists() when
        the removal has to be guaranteed.

    Args:
        directory_path: The path to the directory to delete.

    Raises:
        OSError: If the directory, or any directory beneath it, cannot be read, if the metadata of an entry inside it
            cannot be read, or if an entry inside it cannot be unlinked.
    """
    # Classifies the target itself before descending into it. Both exists() and iterdir() resolve a link, so a link
    # passed as the target would otherwise have the contents of the tree it points at read as its own and deleted.
    # Unlinking it here applies the rule to the argument that the classification below applies to every entry inside it.
    is_symlink, _ = _classify_entry(path=directory_path)
    if is_symlink:
        directory_path.unlink()
        return

    if not directory_path.exists():
        return

    # Classifies entries through lstat, which reports a symlink as a link rather than as its target.
    files = []
    subdirectories = []
    for path in directory_path.iterdir():
        _, is_directory = _classify_entry(path=path)
        if is_directory:
            subdirectories.append(path)
        else:
            files.append(path)

    with ThreadPoolExecutor() as executor:
        list(executor.map(Path.unlink, files))  # Forces completion of all tasks.

    for subdirectory in subdirectories:
        delete_directory(directory_path=subdirectory)

    # Removes the now-empty root directory. Retries with a short delay between attempts, because Windows can be slow
    # to release file handles.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    for _ in range(_MAXIMUM_DELETION_ATTEMPTS):
        try:
            directory_path.rmdir()
            break
        except Exception:
            delay_timer.delay(block=False, delay=_DELETION_RETRY_DELAY_MILLISECONDS, allow_sleep=True)
            continue
    else:
        # Exhausting every attempt leaves a directory that the caller has been told is gone, so the outcome is
        # reported rather than returned silently. It stays a warning, since the data the caller asked to remove is
        # still intact and every caller that needs the removal is able to verify the path itself.
        message = (
            f"Unable to remove the {directory_path} directory after {_MAXIMUM_DELETION_ATTEMPTS} attempts. The "
            f"directory is left in place, which typically indicates that another process still holds an open handle "
            f"to it or to one of the entries it stored."
        )
        console.echo(message=message, level=LogLevel.WARNING)


def transfer_directory(
    source: Path,
    destination: Path,
    num_threads: int = 1,
    *,
    verify_integrity: bool = False,
    remove_source: bool = False,
    progress: bool = False,
    reset_dirty_destination: bool = False,
) -> None:
    """Copies the contents of the input source directory to the destination directory while preserving the underlying
    directory hierarchy.

    Notes:
        Recreates the source hierarchy on the destination before copying any file, and copies with a thread pool when
        ``num_threads`` is greater than 1.

        When integrity verification is enabled, reuses the xxHash3-128 checksum stored in the source directory's
        ax_checksum.txt file when that file exists. Otherwise, generates the checksum before the transfer and writes it
        to the source directory as the ax_checksum.txt file. After the transfer, recomputes the checksum for the
        destination directory and compares it against the source checksum to detect data corruption. A reused checksum
        that predates the source's current contents fails that comparison even though every transferred byte is
        correct, so the failure report names it alongside corruption as a possible cause.

        A symlink is rejected, whether it stands for the source root itself or sits anywhere inside the source tree. A
        link is meaningful only relative to the filesystem that holds it, so moving one is either a silent omission or
        a dangling entry at the destination. A link standing for the root is refused for a second reason. Removing the
        source afterwards would remove the link rather than the tree it names, which leaves the caller told the source
        is gone while every byte of it is still on disk. Every link has to resolve into real data before the tree is
        transferred.

        A destination holding files the source does not account for is also rejected, since the integrity check covers
        the whole destination and those files would fail it while every transferred byte is correct. The checksum file
        never counts as unaccounted, because the transfer overwrites it.

    Args:
        source: The path to the directory to be transferred.
        destination: The path to the destination directory where to move the contents of the source directory.
        num_threads: The number of threads to use for the parallel file transfer. Setting this value below 1 instructs
            the function to use all available CPU cores minus a small number reserved for the host system.
        verify_integrity: Determines whether to perform integrity verification for the transferred files.
        remove_source: Determines whether to remove the source directory after the transfer is complete and
            (optionally) verified.
        progress: Determines whether to track the transfer progress using a progress bar.
        reset_dirty_destination: Determines whether to delete the destination files the source does not account for,
            rather than rejecting the transfer when any are found.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        OSError: If any directory inside the source or the destination tree cannot be read, or if the destination
            path already exists as a file rather than as a directory. Also raised if a copied file cannot be read or
            written, or if the source cannot be removed once ``remove_source`` is enabled. A source discovery failure
            leaves the destination untouched, since the source tree is discovered before anything is written.
        RuntimeError: If the source path is itself a symlink or the source directory contains one, or if
            ``verify_integrity`` is enabled while the source holds no file the checksum can cover. Also raised if the
            destination holds unaccounted files while ``reset_dirty_destination`` is disabled, or if the transferred
            files do not pass the xxHash3-128 checksum integrity verification.
    """
    if not source.exists():
        message = f"Unable to transfer the source directory {source}, as it does not exist."
        console.error(message=message, error=FileNotFoundError)

    # Rejects a link standing for the source root, for the reason the tree-level check below rejects a link inside the
    # tree, and for one more. Removing the source afterwards removes the link rather than the tree it names, since
    # delete_directory applies the same link rule to its argument, so the transfer would report a removal that left
    # every byte of the source on disk.
    if source.is_symlink():
        message = (
            f"Unable to transfer the source directory {source}, as the path is a symbolic link rather than a real "
            f"directory. A link resolves only against the filesystem that holds it, and removing the source after the "
            f"transfer would remove the link while leaving the tree it names in place. Pass the directory the link "
            f"resolves to."
        )
        console.error(message=message, error=RuntimeError)

    if num_threads < 1:
        num_threads = resolve_worker_count()

    # Discovers the source directory contents before anything is written, so a rejected transfer leaves no checksum
    # file behind and creates no destination entries.
    subdirectories, file_list, symlinks = _collect_source_items(source=source)

    if symlinks:
        message = (
            f"Unable to transfer the source directory {source}, as it contains {len(symlinks)} symbolic link(s). A "
            f"link resolves only against the filesystem holding it, so transferring one either drops the data it "
            f"stands for or leaves a dangling entry at the destination. Resolve the following link(s) into real "
            f"data before transferring the tree: {', '.join(str(link.relative_to(source)) for link in symlinks)}."
        )
        console.error(message=message, error=RuntimeError)

    # A verified transfer needs a digest over real data, and the checksum refuses a directory holding no file it can
    # cover. Rejecting here keeps that refusal ahead of every destination write below, so a rejected transfer leaves
    # the destination as it found it. The checksum file is discounted, since the digest excludes it either way.
    if verify_integrity and all(source_file.name == CHECKSUM_FILENAME for source_file in file_list):
        message = (
            f"Unable to transfer the source directory {source} with integrity verification enabled, as the directory "
            f"holds no file the checksum can cover. Disable the 'verify_integrity' flag to transfer a directory tree "
            f"that stores no data."
        )
        console.error(message=message, error=RuntimeError)

    # Declares the destination as a directory, since a destination whose own name carries a dot would otherwise read
    # as a file path and leave only its parent created.
    ensure_directory_exists(path=destination, is_file=False)

    # Reconciles the destination against the source before writing to it. The integrity check hashes the whole
    # destination tree, so a file the source does not account for fails verification even when every transferred byte
    # is correct.
    unaccounted_files = _find_unaccounted_destination_files(
        source=source, destination=destination, source_files=file_list
    )
    if unaccounted_files and not reset_dirty_destination:
        message = (
            f"Unable to transfer the source directory {source} to {destination}, as the destination holds "
            f"{len(unaccounted_files)} file(s) the source does not account for. These files would fail the integrity "
            f"check the transfer performs. Remove them, or enable the 'reset_dirty_destination' flag to have the "
            f"transfer remove them: {', '.join(str(file.relative_to(destination)) for file in unaccounted_files)}."
        )
        console.error(message=message, error=RuntimeError)
    for unaccounted_file in unaccounted_files:
        unaccounted_file.unlink()

    # A checksum written here postdates the discovery above, so it is added to the transfer set explicitly and travels
    # to the destination with the data it covers.
    reused_source_checksum = False
    if verify_integrity:
        source_checksum_path = source.joinpath(CHECKSUM_FILENAME)
        reused_source_checksum = source_checksum_path.exists()
        if not reused_source_checksum:
            calculate_directory_checksum(directory=source, progress=False, save_checksum=True)
            file_list.append(source_checksum_path)

    for destination_directory_path in _plan_destination_directories(
        source=source,
        destination=destination,
        subdirectories=subdirectories,
    ):
        destination_directory_path.mkdir(parents=True, exist_ok=True)

    # I/O operations release the GIL, so a thread pool suffices for the parallel copy and processes are unnecessary.
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    _transfer_file,
                    source_file=file,
                    source_directory=source,
                    destination_directory=destination,
                )
                for file in file_list
            ]
            if progress:
                with console.progress(
                    total=len(file_list),
                    description=f"Transferring files to {destination.name}",
                    unit="file",
                ) as progress_bar:
                    for future in as_completed(futures):
                        future.result()  # Propagates any exceptions from the file transfer.
                        progress_bar.update(n=1)
            else:
                for future in as_completed(futures):
                    future.result()
    elif progress:
        for file in console.track(
            iterable=file_list,
            description=f"Transferring files to {destination.name}",
            unit="file",
        ):
            _transfer_file(source_file=file, source_directory=source, destination_directory=destination)
    else:
        for file in file_list:
            _transfer_file(source_file=file, source_directory=source, destination_directory=destination)

    if verify_integrity:
        destination_checksum = calculate_directory_checksum(directory=destination, progress=False, save_checksum=False)
        with source.joinpath(CHECKSUM_FILENAME).open("r") as local_checksum:
            if destination_checksum != local_checksum.readline().strip():
                # A reused digest describes the source as it stood when that digest was written, so it fails this
                # comparison whenever the source changed since. Naming corruption alone there would send the caller
                # after a transmission fault that did not happen.
                if reused_source_checksum:
                    cause = (
                        f"Either the data was corrupted in transmission, or the reused '{CHECKSUM_FILENAME}' file "
                        f"predates the current contents of the source. Remove that file to have this function "
                        f"regenerate it from the data actually being transferred."
                    )
                else:
                    cause = "This indicates that the data was corrupted in transmission."
                message = (
                    f"Unable to verify the integrity of the directory transferred from {Path(*source.parts[-6:])} to "
                    f"{Path(*destination.parts[-6:])}. The destination checksum must match the source checksum, but "
                    f"the two differ. {cause}"
                )
                console.error(message=message, error=RuntimeError)

    if remove_source:
        message = (
            f"Removing the now-redundant source directory {source} and all of its contents following the successful "
            f"transfer..."
        )
        console.echo(message=message)
        delete_directory(directory_path=source)


def _collect_source_items(source: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Discovers the contents of the source directory and separates them into subdirectories, files, and symlinks.

    Notes:
        Every list is sorted by path depth so that parent directories precede their children and the file copy order
        is deterministic. Any item that is neither a directory nor a symlink is treated as a file.

        Symlinks are reported separately, since a link stands for data living outside the tree the caller named.

    Args:
        source: The root directory whose contents are discovered.

    Returns:
        The subdirectories, files, and symlinks found anywhere under the source directory, each sorted by path depth.

    Raises:
        OSError: If the source directory does not exist, is not a directory, or cannot be read, if any directory
            beneath it cannot be read, or if the metadata of an entry beneath it cannot be read.
    """
    all_items = sorted(walk_directory(directory=source), key=lambda path: len(path.relative_to(source).parts))

    symlinks: list[Path] = []
    subdirectories: list[Path] = []
    files: list[Path] = []
    for item in all_items:
        is_symlink, is_directory = _classify_entry(path=item)
        if is_symlink:
            symlinks.append(item)
        elif is_directory:
            subdirectories.append(item)
        else:
            files.append(item)

    return subdirectories, files, symlinks


def _classify_entry(path: Path) -> tuple[bool, bool]:
    """Determines whether the target filesystem entry is a symbolic link and whether it is a directory.

    Notes:
        One lstat call answers both questions. Since lstat reports a link rather than its target, a symbolic link
        answers False to the directory question whatever it points at.

        An entry whose metadata query fails in a way ``reports_absent_entry`` accepts answers False to both questions,
        which covers an entry that disappears between its discovery and this call. Every other failure propagates,
        since an entry that exists but cannot be read has an unknown kind rather than no kind.

    Args:
        path: The filesystem entry to classify.

    Returns:
        The first element is True when the entry is a symbolic link. The second is True when the entry is a directory.

    Raises:
        OSError: If the entry's metadata cannot be read for any reason other than the entry being absent.
    """
    try:
        entry_mode = os.lstat(path).st_mode
    except OSError as error:
        if not reports_absent_entry(error=error):
            raise
        return False, False
    return stat.S_ISLNK(entry_mode), stat.S_ISDIR(entry_mode)


def _find_unaccounted_destination_files(source: Path, destination: Path, source_files: list[Path]) -> list[Path]:
    """Finds the destination files the source directory does not account for.

    Notes:
        The checksum file never counts as unaccounted, because the transfer overwrites it with the source's own copy.
        Directories are ignored, since an empty directory contributes nothing to the integrity check.

    Args:
        source: The root source directory the transferred files are relative to.
        destination: The destination directory to reconcile against the source.
        source_files: The files the transfer copies out of the source directory.

    Returns:
        The unaccounted destination files, sorted by path.

    Raises:
        OSError: If the destination directory does not exist, is not a directory, or cannot be read, if any
            directory beneath it cannot be read, or if the kind of an entry beneath it cannot be determined.
    """
    expected = {file_path.relative_to(source) for file_path in source_files}
    return sorted(
        path
        for path in walk_files(directory=destination)
        if path.name != CHECKSUM_FILENAME and path.relative_to(destination) not in expected
    )


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


def _transfer_file(source_file: Path, source_directory: Path, destination_directory: Path) -> None:
    """Copies the input file from the source directory to the destination directory while preserving the file metadata.

    Notes:
        If the file is found under a hierarchy of subdirectories inside the input ``source_directory``, that hierarchy
        is preserved in the destination directory.

    Args:
        source_file: The file to be copied.
        source_directory: The root directory where the file is located.
        destination_directory: The destination directory where to move the file.
    """
    relative = source_file.relative_to(source_directory)
    destination_file = destination_directory / relative
    shutil.copy2(src=source_file, dst=destination_file)
