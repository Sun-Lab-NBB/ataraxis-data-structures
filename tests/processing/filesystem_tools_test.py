"""Contains tests for the filesystem_tools module provided by the processing package."""

import os
import errno
from typing import Any
from pathlib import Path
from collections.abc import Iterator

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import transfer_directory, calculate_directory_checksum
from ataraxis_data_structures.processing.checksum_tools import _discover_checksum_files
from ataraxis_data_structures.processing.filesystem_tools import (
    walk_files,
    walk_directory,
    resolve_unique_roots,
    discover_marker_files,
    discover_marker_roots,
)

_REQUIRES_ENFORCED_PERMISSIONS: pytest.MarkDecorator = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="Directory permission bits are enforced only on POSIX hosts running as an unprivileged user.",
)
"""Limits a test to the hosts that enforce a directory's permission bits.

Notes:
    An injected scan failure covers the unreadable-directory case on every platform, but it lands upstream of the
    per-entry metadata query. A test that needs the second condition builds it from a real permission bit, which
    Windows ignores on a directory and which a privileged user bypasses everywhere.
"""


@pytest.fixture
def unsearchable_tree(tmp_path: Path) -> Iterator[Path]:
    """Builds a directory tree whose one subdirectory can be listed but whose entries cannot be inspected."""
    root = tmp_path / "tree"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (root / "visible.txt").write_text("visible")
    (locked / "hidden.txt").write_text("hidden")
    locked.chmod(0o444)

    yield root

    locked.chmod(0o700)


@pytest.fixture
def unreadable_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Builds a directory tree holding one readable file and one subdirectory whose scan is refused.

    The refusal is injected rather than produced with a permission bit, so the tests consuming this fixture run on
    every supported platform.
    """
    root = tmp_path / "tree"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (root / "visible.txt").write_text("visible")
    (locked / "hidden.txt").write_text("hidden")

    real_scandir = os.scandir

    def _refuse_locked_directory(path: Any) -> Any:
        # Compares the rendered paths, so a scan the interpreter performs against a descriptor passes through
        # untouched.
        if str(path) == str(locked):
            raise PermissionError(errno.EACCES, "Permission denied", str(locked))
        return real_scandir(path)

    monkeypatch.setattr(target=os, name="scandir", value=_refuse_locked_directory)
    return root


def test_walk_directory_discovers_every_entry(tmp_path: Path) -> None:
    """Verifies that walk_directory reports files, subdirectories, and links found at any depth."""
    (tmp_path / "top.txt").write_text("top")
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / "nested" / "inner.txt").write_text("inner")
    (tmp_path / "nested" / "deeper" / "deepest.txt").write_text("deepest")
    (tmp_path / "link").symlink_to(tmp_path / "top.txt")

    discovered = {path.relative_to(tmp_path).as_posix() for path in walk_directory(directory=tmp_path)}

    assert discovered == {"top.txt", "nested", "nested/deeper", "nested/inner.txt", "nested/deeper/deepest.txt", "link"}


def test_walk_directory_does_not_follow_links_to_directories(tmp_path: Path) -> None:
    """Verifies that walk_directory reports a link to a directory without descending into its target."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    discovered = {path.relative_to(root).as_posix() for path in walk_directory(directory=root)}

    assert discovered == {"link"}


def test_walk_directory_returns_nothing_for_an_empty_directory(tmp_path: Path) -> None:
    """Verifies that walk_directory reports no entry for a directory that holds none."""
    assert walk_directory(directory=tmp_path) == []


def test_walk_directory_raises_for_a_missing_directory(tmp_path: Path) -> None:
    """Verifies that walk_directory raises when the root directory does not exist."""
    with pytest.raises(FileNotFoundError):
        walk_directory(directory=tmp_path / "never_created")


def test_walk_directory_raises_for_an_unreadable_subdirectory(unreadable_tree: Path) -> None:
    """Verifies that walk_directory raises instead of silently omitting a subdirectory it cannot read."""
    with pytest.raises(PermissionError):
        walk_directory(directory=unreadable_tree)


def test_walk_files_reports_files_and_omits_directories(tmp_path: Path) -> None:
    """Verifies that walk_files reports the files found at any depth without reporting the directories holding them."""
    (tmp_path / "top.txt").write_text("top")
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / "nested" / "inner.txt").write_text("inner")

    discovered = {path.relative_to(tmp_path).as_posix() for path in walk_files(directory=tmp_path)}

    assert discovered == {"top.txt", "nested/inner.txt"}


def test_walk_files_resolves_links_to_decide_the_kind(tmp_path: Path) -> None:
    """Verifies that walk_files reports a link to a file and omits a link to a directory."""
    target = tmp_path / "target.txt"
    target.write_text("target")
    (tmp_path / "target_directory").mkdir()
    (tmp_path / "link_to_file").symlink_to(target)
    (tmp_path / "link_to_directory").symlink_to(tmp_path / "target_directory", target_is_directory=True)

    discovered = {path.relative_to(tmp_path).as_posix() for path in walk_files(directory=tmp_path)}

    assert discovered == {"target.txt", "link_to_file"}


def test_walk_files_reports_no_file_for_a_link_that_resolves_to_nothing(tmp_path: Path) -> None:
    """Verifies that walk_files omits every unresolvable link kind without failing the traversal."""
    (tmp_path / "data.txt").write_text("payload")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    (tmp_path / "loop_a").symlink_to(tmp_path / "loop_b")
    (tmp_path / "loop_b").symlink_to(tmp_path / "loop_a")
    (tmp_path / "through_file").symlink_to(tmp_path / "data.txt" / "child")

    # The three unresolvable kinds answer alike rather than one of them aborting the whole traversal.
    discovered = {path.relative_to(tmp_path).as_posix() for path in walk_files(directory=tmp_path)}

    assert discovered == {"data.txt"}


def test_walk_files_propagates_a_kind_failure_that_is_not_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that walk_files raises when an entry's kind query is denied by permission."""
    entry = _RefusingEntry(path=tmp_path / "entry.bin", error_number=errno.EACCES)
    monkeypatch.setattr(target=os, name="scandir", value=lambda _path: _StubScan(entries=[entry]))

    with pytest.raises(PermissionError):
        walk_files(directory=tmp_path)


def test_walk_files_tolerates_a_kind_failure_that_means_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that walk_files omits an entry whose kind query reports it as absent."""
    entry = _RefusingEntry(path=tmp_path / "entry.bin", error_number=errno.ENOENT)
    monkeypatch.setattr(target=os, name="scandir", value=lambda _path: _StubScan(entries=[entry]))

    assert walk_files(directory=tmp_path) == []


def test_walk_files_tolerates_the_windows_link_loop_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that walk_files omits an entry whose kind query reports the Windows link-loop status."""
    # Windows files the condition under the generic EINVAL, which every other caller has to keep propagating, so the
    # status code is the only part of the failure that marks the entry as unresolvable.
    entry = _RefusingEntry(path=tmp_path / "loop_a", error_number=errno.EINVAL, error_type=_WindowsLinkLoopError)
    monkeypatch.setattr(target=os, name="scandir", value=lambda _path: _StubScan(entries=[entry]))

    assert walk_files(directory=tmp_path) == []


def test_walk_files_propagates_an_invalid_argument_failure_without_the_link_loop_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that walk_files raises for an EINVAL kind query that does not carry the link-loop status."""
    entry = _RefusingEntry(path=tmp_path / "entry.bin", error_number=errno.EINVAL, error_type=OSError)
    monkeypatch.setattr(target=os, name="scandir", value=lambda _path: _StubScan(entries=[entry]))

    with pytest.raises(OSError, match="Injected failure"):
        walk_files(directory=tmp_path)


def test_walk_files_raises_for_an_unreadable_subdirectory(unreadable_tree: Path) -> None:
    """Verifies that walk_files raises instead of silently omitting a subdirectory it cannot read."""
    with pytest.raises(PermissionError):
        walk_files(directory=unreadable_tree)


def test_checksum_discovery_does_not_route_through_the_path_file_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that checksum discovery reports a file whose path-level file test answers False."""
    (tmp_path / "visible.txt").write_text("visible")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.txt").write_text("hidden")

    # Path.is_file() answers False for an entry it cannot stat on some interpreters, which would drop that entry from
    # the digest and certify a subset of the tree as the whole. Discovery reads the kind the directory scan returned.
    monkeypatch.setattr(target=Path, name="is_file", value=lambda _self: False)

    discovered = {
        path.relative_to(tmp_path).as_posix()
        for path in _discover_checksum_files(directory=tmp_path, excluded_files=set())
    }

    assert discovered == {"visible.txt", "nested/hidden.txt"}


def test_calculate_directory_checksum_raises_for_an_unreadable_subdirectory(unreadable_tree: Path) -> None:
    """Verifies that a directory checksum fails rather than certifying the subset of the tree it can read."""
    with pytest.raises(PermissionError):
        calculate_directory_checksum(directory=unreadable_tree, progress=False, save_checksum=False)


@_REQUIRES_ENFORCED_PERMISSIONS
def test_calculate_directory_checksum_raises_for_an_unsearchable_subdirectory(unsearchable_tree: Path) -> None:
    """Verifies that a directory checksum fails for a tree whose entries cannot be inspected."""
    with pytest.raises(PermissionError):
        calculate_directory_checksum(directory=unsearchable_tree, progress=False, save_checksum=False)


def test_transfer_directory_raises_for_an_unreadable_source_subdirectory(unreadable_tree: Path, tmp_path: Path) -> None:
    """Verifies that a transfer fails rather than silently copying the subset of the source it can read."""
    destination = tmp_path / "destination"
    destination.mkdir()
    stale = destination / "stale.txt"
    stale.write_text("must survive a rejected transfer")

    with pytest.raises(PermissionError):
        transfer_directory(
            source=unreadable_tree,
            destination=destination,
            verify_integrity=True,
            reset_dirty_destination=True,
        )

    # The transfer discovers the source before it writes, so a rejected transfer leaves the destination as it found it.
    assert stale.exists()
    assert not (unreadable_tree / "ax_checksum.txt").exists()


@_REQUIRES_ENFORCED_PERMISSIONS
def test_transfer_directory_rejects_an_unaccounted_file_under_an_unsearchable_destination_directory(
    tmp_path: Path,
) -> None:
    """Verifies that a transfer into a destination holding an unaccounted file it cannot inspect is refused."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("payload")
    destination = tmp_path / "destination"
    locked = destination / "locked"
    locked.mkdir(parents=True)
    (locked / "stray.txt").write_text("stray")
    locked.chmod(0o444)

    try:
        with pytest.raises((RuntimeError, PermissionError)):
            transfer_directory(source=source, destination=destination, verify_integrity=False)
    finally:
        locked.chmod(0o700)


def test_transfer_directory_rejects_a_verified_source_holding_no_coverable_file(tmp_path: Path) -> None:
    """Verifies that a verified transfer of a file-less source is refused before the destination is written to."""
    source = tmp_path / "source"
    (source / "empty_subdirectory").mkdir(parents=True)
    destination = tmp_path / "destination"
    destination.mkdir()
    stray = destination / "stray.txt"
    stray.write_text("destination data")

    with pytest.raises(RuntimeError, match="holds no file the checksum can cover"):
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
            reset_dirty_destination=True,
        )

    # The refusal precedes every destination write, so the unaccounted file the transfer would otherwise delete stays.
    assert stray.exists()
    assert not (source / "ax_checksum.txt").exists()


def test_transfer_directory_rejects_a_verified_source_holding_only_a_checksum_file(tmp_path: Path) -> None:
    """Verifies that a source whose only file is the checksum file is refused, since the digest excludes that file."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "ax_checksum.txt").write_text("stale")
    destination = tmp_path / "destination"

    with pytest.raises(RuntimeError, match="holds no file the checksum can cover"):
        transfer_directory(source=source, destination=destination, verify_integrity=True)


def test_transfer_directory_still_transfers_a_file_less_source_without_verification(tmp_path: Path) -> None:
    """Verifies that a source holding no file transfers normally while integrity verification is disabled."""
    source = tmp_path / "source"
    (source / "empty_subdirectory").mkdir(parents=True)
    destination = tmp_path / "destination"

    transfer_directory(source=source, destination=destination, verify_integrity=False)

    assert (destination / "empty_subdirectory").is_dir()


def test_discover_marker_files_reports_matches_at_any_depth(tmp_path: Path) -> None:
    """Verifies that marker discovery reports every matching file in the tree, sorted by path."""
    (tmp_path / "animal_2" / "session_b" / "raw_data").mkdir(parents=True)
    (tmp_path / "animal_1" / "session_a" / "raw_data").mkdir(parents=True)
    first = tmp_path / "animal_1" / "session_a" / "raw_data" / "session_data.yaml"
    second = tmp_path / "animal_2" / "session_b" / "raw_data" / "session_data.yaml"
    first.write_text("first")
    second.write_text("second")
    (tmp_path / "animal_1" / "other.yaml").write_text("other")

    assert discover_marker_files(directory=tmp_path, marker_name="session_data.yaml") == [first, second]


def test_discover_marker_files_omits_a_directory_carrying_the_marker_name(tmp_path: Path) -> None:
    """Verifies that a directory named after the marker is not reported as a marker."""
    (tmp_path / "session_data.yaml").mkdir()
    (tmp_path / "nested").mkdir()
    marker = tmp_path / "nested" / "session_data.yaml"
    marker.write_text("payload")

    assert discover_marker_files(directory=tmp_path, marker_name="session_data.yaml") == [marker]


def test_discover_marker_files_reports_nothing_when_no_marker_matches(tmp_path: Path) -> None:
    """Verifies that a tree holding no matching marker yields an empty result."""
    (tmp_path / "data.txt").write_text("payload")

    assert discover_marker_files(directory=tmp_path, marker_name="session_data.yaml") == []


def test_discover_marker_files_raises_for_an_unreadable_subdirectory(unreadable_tree: Path) -> None:
    """Verifies that marker discovery raises instead of silently omitting a subdirectory it cannot read."""
    with pytest.raises(PermissionError):
        discover_marker_files(directory=unreadable_tree, marker_name="hidden.txt")


def test_discover_marker_roots_resolves_the_marker_parent_by_default(tmp_path: Path) -> None:
    """Verifies that a marker written into the directory it describes resolves to that directory."""
    owner = tmp_path / "dataset"
    owner.mkdir()
    (owner / "dataset.yaml").write_text("payload")

    assert discover_marker_roots(directory=tmp_path, marker_name="dataset.yaml") == [owner]


def test_discover_marker_roots_climbs_the_requested_number_of_levels(tmp_path: Path) -> None:
    """Verifies that a marker nested below its owner resolves to the ancestor at the requested level."""
    owner = tmp_path / "animal_1" / "session_a"
    (owner / "raw_data").mkdir(parents=True)
    (owner / "raw_data" / "session_data.yaml").write_text("payload")

    assert discover_marker_roots(directory=tmp_path, marker_name="session_data.yaml", levels_up=1) == [owner]


def test_discover_marker_roots_reports_one_entry_per_owning_directory(tmp_path: Path) -> None:
    """Verifies that two markers resolving to the same owner contribute a single entry."""
    owner = tmp_path / "session"
    (owner / "raw_data").mkdir(parents=True)
    (owner / "processed_data").mkdir()
    (owner / "raw_data" / "marker.yaml").write_text("first")
    (owner / "processed_data" / "marker.yaml").write_text("second")

    assert discover_marker_roots(directory=tmp_path, marker_name="marker.yaml", levels_up=1) == [owner]


def test_discover_marker_roots_rejects_a_negative_level_count(tmp_path: Path) -> None:
    """Verifies that a negative level count is rejected."""
    message = (
        f"Unable to discover the roots of the 'marker.yaml' markers stored under '{tmp_path}'. The 'levels_up' "
        f"argument must be greater than or equal to 0, but got -1."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        discover_marker_roots(directory=tmp_path, marker_name="marker.yaml", levels_up=-1)


def test_discover_marker_roots_rejects_a_marker_without_the_requested_ancestor(tmp_path: Path) -> None:
    """Verifies that a marker sitting too close to the filesystem root to have the requested ancestor is rejected."""
    marker = tmp_path / "marker.yaml"
    marker.write_text("payload")

    with pytest.raises(ValueError, match="ancestor"):
        discover_marker_roots(directory=tmp_path, marker_name="marker.yaml", levels_up=len(marker.parents))


def test_resolve_unique_roots_truncates_each_path_at_its_distinguishing_component() -> None:
    """Verifies that paths sharing a layout resolve to the ancestors naming what each one belongs to."""
    first = Path("/data/animal_1/2026-01-01/raw_data/behavior_data_log")
    second = Path("/data/animal_2/2026-02-02/raw_data/behavior_data_log")

    assert resolve_unique_roots(paths=[first, second]) == (
        Path("/data/animal_1/2026-01-01"),
        Path("/data/animal_2/2026-02-02"),
    )


def test_resolve_unique_roots_resolves_a_lone_path_to_itself() -> None:
    """Verifies that a path with no sibling to differ from resolves to its own final component."""
    path = Path("/data/animal_1/raw_data")

    assert resolve_unique_roots(paths=(path,)) == (path,)


def test_resolve_unique_roots_reports_one_entry_per_distinct_root() -> None:
    """Verifies that sibling paths distinguished by their own final components each resolve to themselves."""
    first = Path("/data/recording_1/logs/camera")
    second = Path("/data/recording_1/logs/microcontroller")

    assert resolve_unique_roots(paths=[first, second]) == (first, second)


def test_resolve_unique_roots_handles_paths_of_differing_depths() -> None:
    """Verifies that resolution assumes no fixed depth for the structure shared below the distinguishing component."""
    first = Path("/data/recording_1/deeply/nested/logs")
    second = Path("/data/recording_2/logs")

    # Each path truncates at its own deepest distinguishing component, so the two resolve at unequal depths.
    assert resolve_unique_roots(paths=[first, second]) == (
        Path("/data/recording_1/deeply/nested"),
        Path("/data/recording_2"),
    )


def test_resolve_unique_roots_rejects_paths_sharing_every_component() -> None:
    """Verifies that a path carrying no component the other paths lack is rejected."""
    path = Path("/data/recording")

    with pytest.raises(ValueError, match="shares all"):
        resolve_unique_roots(paths=[path, path])


def test_resolve_unique_roots_stops_at_the_filesystem_root() -> None:
    """Verifies that a distinguishing component the upward walk cannot match terminates the walk at the root."""
    # The absolute path is distinguished by its root component alone, which no ancestor carries as its name, so the
    # walk climbs to the filesystem root rather than looping there.
    absolute = Path("/shared")
    relative = Path("relative/shared")

    assert resolve_unique_roots(paths=[absolute, relative]) == (Path(absolute.parts[0]), Path("relative"))


class _StubScan:
    """Stands in for the scandir context manager, yielding the entries the test supplies.

    Attributes:
        _entries: Cached entries the context manager yields.
    """

    def __init__(self, entries: list[Any]) -> None:
        self._entries = entries

    def __enter__(self) -> Iterator[Any]:
        return iter(self._entries)

    def __exit__(self, *_arguments: object) -> bool:
        return False


class _WindowsLinkLoopError(OSError):
    """Stands in for the failure Windows raises for a link chain that does not resolve.

    Notes:
        Windows carries the condition in the read-only 'winerror' attribute, which POSIX hosts do not define at all.
        Shadowing it with a plain class attribute makes the injected failure read alike on every platform, so the
        status the traversal reads is exercised wherever the suite runs.
    """

    winerror = 1921


class _RefusingEntry:
    """Stands in for a scan entry whose kind query fails with the error the test supplies.

    Attributes:
        path: The rendered path of the stand-in entry.
        _error_number: Cached errno the kind query fails with.
        _error_type: Cached exception class the kind query raises.
    """

    def __init__(self, path: Path, error_number: int, error_type: type[OSError] = PermissionError) -> None:
        self.path = str(path)
        self._error_number = error_number
        self._error_type = error_type

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:  # noqa: ARG002 - The stub entry is never a directory.
        return False

    def is_file(self) -> bool:
        raise self._error_type(self._error_number, "Injected failure", self.path)
