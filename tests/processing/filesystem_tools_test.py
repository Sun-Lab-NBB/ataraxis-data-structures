"""Contains tests for the filesystem_tools module provided by the processing package."""

import os
import errno
from pathlib import Path

import pytest

from ataraxis_data_structures import transfer_directory, calculate_directory_checksum
from ataraxis_data_structures.processing.checksum_tools import _discover_checksum_files
from ataraxis_data_structures.processing.filesystem_tools import walk_files, walk_directory

# An injected scan failure covers the unreadable-directory case on every platform, but it lands upstream of the
# per-entry metadata query, so the tests below build that second condition from a real permission bit instead. Those
# bits are enforced only on a POSIX host running as an unprivileged user.
requires_enforced_permissions = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="Directory permission bits are enforced only on POSIX hosts running as an unprivileged user.",
)


@pytest.fixture
def unsearchable_tree(tmp_path: Path) -> Path:
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

    The refusal is injected rather than produced with a permission bit. A directory's read permission is a no-op on
    Windows and is bypassed by a privileged user on every platform, so a permission bit would leave these tests
    unexercised on part of the supported matrix.
    """
    root = tmp_path / "tree"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (root / "visible.txt").write_text("visible")
    (locked / "hidden.txt").write_text("hidden")

    real_scandir = os.scandir

    def _refuse_locked_directory(path):
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


class _StubScan:
    """Stands in for the scandir context manager, yielding the entries the test supplies."""

    def __init__(self, entries):
        self._entries = entries

    def __enter__(self):
        return iter(self._entries)

    def __exit__(self, *_arguments):
        return False


class _RefusingEntry:
    """Stands in for a scan entry whose kind query fails with the errno the test supplies."""

    def __init__(self, path, error_number):
        self.path = str(path)
        self._error_number = error_number

    def is_dir(self, *, follow_symlinks=True):  # noqa: ARG002 - The stub entry is never a directory.
        return False

    def is_file(self):
        raise PermissionError(self._error_number, "Injected failure", self.path)


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


def test_walk_files_reports_no_file_for_a_link_that_resolves_to_nothing(tmp_path: Path) -> None:
    """Verifies that walk_files omits every unresolvable link kind without failing the traversal."""
    (tmp_path / "data.txt").write_text("payload")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    (tmp_path / "loop_a").symlink_to(tmp_path / "loop_b")
    (tmp_path / "loop_b").symlink_to(tmp_path / "loop_a")
    (tmp_path / "through_file").symlink_to(tmp_path / "data.txt" / "child")

    # A link that resolves to nothing is no file, and the three kinds answer alike rather than one of them aborting
    # the whole traversal.
    discovered = {path.relative_to(tmp_path).as_posix() for path in walk_files(directory=tmp_path)}

    assert discovered == {"data.txt"}


@requires_enforced_permissions
def test_calculate_directory_checksum_raises_for_an_unsearchable_subdirectory(unsearchable_tree: Path) -> None:
    """Verifies that a directory checksum fails for a tree whose entries cannot be inspected."""
    with pytest.raises(PermissionError):
        calculate_directory_checksum(directory=unsearchable_tree, progress=False, save_checksum=False)


@requires_enforced_permissions
def test_transfer_directory_rejects_an_unaccounted_file_under_an_unsearchable_destination_directory(
    tmp_path: Path,
) -> None:
    """Verifies that a destination file the source does not account for is found even when it cannot be inspected."""
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
