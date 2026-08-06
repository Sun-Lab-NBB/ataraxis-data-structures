"""Provides assets for traversing filesystem directory trees."""

import os
import errno
from pathlib import Path

ABSENT_ENTRY_ERRNOS: frozenset[int] = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})
"""The metadata-query failures that answer as an absent entry rather than propagating.

Notes:
    An entry that disappears between its discovery and its inspection has to read as absent, since discovery and
    inspection are separate filesystem calls. ENOENT, ENOTDIR, and ELOOP each name a path component that does not
    resolve, and EBADF joins them to absorb the spurious failure macOS stat raises.

    Every other failure, a permission error above all, means the entry exists while its kind stays unknown. Answering
    that case as absent would drop the entry from a result meant to cover the whole tree.
"""


def walk_directory(directory: Path) -> list[Path]:
    """Discovers every entry stored anywhere under the target directory.

    Notes:
        Symbolic links are reported without being followed, so the traversal stays inside the tree the caller named.

    Args:
        directory: The root directory whose contents are discovered.

    Returns:
        Every subdirectory, file, and link found anywhere under the root directory, in an unspecified order.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
            beneath it cannot be read.
    """
    return [Path(entry.path) for entry in _scan_tree(directory=directory)]


def walk_files(directory: Path) -> list[Path]:
    """Discovers every entry stored anywhere under the target directory that resolves to a file.

    Notes:
        The kind of a regular entry comes from the record its directory scan returned, so an entry whose own metadata
        query is denied is still reported. Deciding the kind through a path instead would answer False for such an
        entry on some interpreters, which would silently narrow the result to the readable subset.

        A link is followed to decide the answer, so a link to a file is reported and a link to a directory is not. A
        link that resolves to nothing is reported as no file, which covers a dangling link, a link chain that loops,
        and a link whose path runs through a file.

    Args:
        directory: The root directory whose contents are discovered.

    Returns:
        Every file found anywhere under the root directory, in an unspecified order.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, if any directory beneath
            it cannot be read, or if the kind of an entry beneath it cannot be determined.
    """
    return [Path(entry.path) for entry in _scan_tree(directory=directory) if _resolves_to_file(entry=entry)]


def _scan_tree(directory: Path) -> list[os.DirEntry[str]]:
    """Scans the target directory and every directory beneath it, collecting the entries each scan returns.

    Notes:
        A directory the process is unable to read raises instead of contributing nothing to the result. The pathlib
        globbing helpers suppress that failure, which silently narrows the tree to the part the process happens to be
        able to read, and leaves every caller believing it covered the whole tree.

        Each returned entry carries the kind its own scan reported, so a caller testing whether an entry is a
        directory pays no additional metadata call on the platforms that supply that record.

    Args:
        directory: The root directory whose tree is scanned.

    Returns:
        The scan entries for everything found anywhere under the root directory, in an unspecified order.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
            beneath it cannot be read.
    """
    entries: list[os.DirEntry[str]] = []
    pending: list[Path] = [directory]

    while pending:
        with os.scandir(pending.pop()) as scan:
            for entry in scan:
                entries.append(entry)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))

    return entries


def _resolves_to_file(entry: os.DirEntry[str]) -> bool:
    """Determines whether the target scan entry resolves to a file, following it when it is a link.

    Args:
        entry: The scan entry whose kind is determined.

    Returns:
        True when the entry resolves to a file, and False when it resolves to anything else or resolves to nothing.

    Raises:
        OSError: If the entry's kind cannot be determined for any reason other than the entry being absent.
    """
    try:
        return entry.is_file()
    except OSError as error:
        if error.errno not in ABSENT_ENTRY_ERRNOS:
            raise
        return False
