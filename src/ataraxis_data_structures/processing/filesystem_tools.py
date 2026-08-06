"""Provides assets for traversing filesystem directory trees and for discovering the data assets stored inside them."""

from __future__ import annotations

import os
import errno
from typing import TYPE_CHECKING
from pathlib import Path

from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def discover_marker_files(directory: Path, marker_name: str) -> list[Path]:
    """Discovers every marker file with the target name stored anywhere under the target directory.

    Notes:
        The name is compared against the entry each directory scan returned, so the traversal keeps only the matches
        rather than materializing every file beneath the root. That bound matters over a tree holding a large number
        of data files alongside the few markers describing them.

    Args:
        directory: The root directory whose tree is searched.
        marker_name: The exact filename every discovered marker carries.

    Returns:
        The paths to every matching file found anywhere under the root directory, sorted by path.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
            beneath it cannot be read.
    """
    return sorted(
        Path(entry.path)
        for entry in _scan_tree(directory=directory)
        if entry.name == marker_name and _resolves_to_file(entry=entry)
    )


def discover_marker_roots(directory: Path, marker_name: str, levels_up: int = 0) -> list[Path]:
    """Discovers the directories owning every marker file stored anywhere under the target directory.

    Notes:
        A marker describes the asset that owns it, and that asset's directory sits a fixed number of levels above the
        marker rather than at the marker itself. A marker written at ``{root}/raw_data/session_data.yaml`` resolves
        its root with one level, while a marker written directly into the directory it describes resolves with none.

        Two markers resolving to the same directory contribute one entry, since the result names owning directories
        rather than the markers that pointed at them.

    Args:
        directory: The root directory whose tree is searched.
        marker_name: The exact filename every discovered marker carries.
        levels_up: The number of directory levels between a marker's own parent and the directory that owns it.

    Returns:
        The paths to every directory owning a matching marker, sorted by path.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
            beneath it cannot be read.
        ValueError: If the requested level count is negative, or if a discovered marker sits too close to the
            filesystem root to have an ancestor at that level.
    """
    if levels_up < 0:
        message = (
            f"Unable to discover the roots of the '{marker_name}' markers stored under '{directory}'. The "
            f"'levels_up' argument must be greater than or equal to 0, but got {levels_up}."
        )
        console.error(message=message, error=ValueError)

    roots: set[Path] = set()
    for marker in discover_marker_files(directory=directory, marker_name=marker_name):
        ancestors = marker.parents
        if levels_up >= len(ancestors):
            message = (
                f"Unable to discover the roots of the '{marker_name}' markers stored under '{directory}'. Resolving "
                f"the root of the marker at '{marker}' requires an ancestor {levels_up} level(s) above its own "
                f"parent, but the marker has only {len(ancestors)} ancestor(s)."
            )
            console.error(message=message, error=ValueError)
        roots.add(ancestors[levels_up])

    return sorted(roots)


def resolve_unique_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolves the target paths to the deepest ancestor of each whose name no other path carries.

    Notes:
        Paths sharing a structural layout differ only in the components naming the asset each one belongs to, such as
        a recording or a session identifier. Truncating each path at its deepest distinguishing component therefore
        strips the structure shared below that component without assuming a fixed depth for it.

        A lone path has no sibling to differ from, so its own final component distinguishes it and it resolves to
        itself.

    Args:
        paths: The paths to resolve. Every path must carry at least one component that no other path carries.

    Returns:
        The resolved ancestors, one per distinct root, in the order the first path resolving to each one appears.

    Raises:
        ValueError: If any path shares every one of its components with the other paths.
    """
    targets = list(paths)
    unique_components = _extract_unique_components(paths=targets)

    roots: list[Path] = []
    for path, unique_component in zip(targets, unique_components, strict=True):
        # Walks up from the path to the ancestor carrying its distinguishing component, stopping at the filesystem
        # root so a component the walk cannot reach terminates the loop rather than spinning on it.
        current = path
        while current.name != unique_component and current != current.parent:
            current = current.parent
        if current not in roots:
            roots.append(current)

    return tuple(roots)


def _extract_unique_components(paths: list[Path]) -> tuple[str, ...]:
    """Extracts the deepest component of each target path that no other target path carries.

    Args:
        paths: The paths whose distinguishing components are extracted.

    Returns:
        One distinguishing component per input path, in input order.

    Raises:
        ValueError: If any path shares every one of its components with the other paths.
    """
    components: list[str] = []
    for index, path in enumerate(paths):
        shared = {
            component for other_index, other in enumerate(paths) if other_index != index for component in other.parts
        }
        unique_component = next((component for component in reversed(path.parts) if component not in shared), None)

        if unique_component is None:
            message = (
                f"Unable to extract the distinguishing component of the path '{path}'. Every path must carry at "
                f"least one component that no other path carries, but this path shares all {len(path.parts)} of its "
                f"components with the others."
            )
            console.error(message=message, error=ValueError)

        components.append(unique_component)

    return tuple(components)


def _scan_tree(directory: Path) -> Iterator[os.DirEntry[str]]:
    """Scans the target directory and every directory beneath it, collecting the entries each scan returns.

    Notes:
        A directory the process is unable to read raises instead of contributing nothing to the result. The pathlib
        globbing helpers suppress that failure, which silently narrows the tree to the part the process happens to be
        able to read, and leaves every caller believing it covered the whole tree.

        Each returned entry carries the kind its own scan reported, so a caller testing whether an entry is a
        directory pays no additional metadata call on the platforms that supply that record.

        Entries are yielded as each scan produces them, so a caller keeping only the entries it selects never holds
        the whole tree at once.

    Args:
        directory: The root directory whose tree is scanned.

    Yields:
        The scan entry for everything found anywhere under the root directory, in an unspecified order.

    Raises:
        OSError: If the root directory does not exist, is not a directory, or cannot be read, or if any directory
            beneath it cannot be read.
    """
    pending: list[Path] = [directory]

    while pending:
        with os.scandir(pending.pop()) as scan:
            for entry in scan:
                yield entry
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


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
