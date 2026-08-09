"""Contains tests for the write_tools module provided by the processing package."""

import os
import sys
from pathlib import Path

import pytest

from ataraxis_data_structures import atomic_write, direct_write
from ataraxis_data_structures.processing import write_tools
from ataraxis_data_structures.processing.write_tools import _RENAME_RETRY_COUNT, _publish_file


def test_atomic_write_publishes_text_contents(tmp_path: Path) -> None:
    """Verifies that the written text reaches the destination once the context exits."""
    file_path = tmp_path / "document.txt"

    with atomic_write(file_path=file_path) as file:
        file.write("payload")

    assert file_path.read_text() == "payload"


def test_atomic_write_publishes_binary_contents(tmp_path: Path) -> None:
    """Verifies that the binary mode accepts bytes and publishes them unchanged."""
    file_path = tmp_path / "document.bin"

    with atomic_write(file_path=file_path, binary=True) as file:
        file.write(b"\x00\x01\x02")

    assert file_path.read_bytes() == b"\x00\x01\x02"


def test_atomic_write_replaces_an_existing_file(tmp_path: Path) -> None:
    """Verifies that a destination holding earlier contents is replaced by the new contents."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("previous")

    with atomic_write(file_path=file_path) as file:
        file.write("current")

    assert file_path.read_text() == "current"


def test_atomic_write_keeps_the_destination_until_the_context_exits(tmp_path: Path) -> None:
    """Verifies that a reader observes the earlier contents while the replacement is still being written."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("previous")

    with atomic_write(file_path=file_path) as file:
        file.write("current")
        assert file_path.read_text() == "previous"

    assert file_path.read_text() == "current"


def test_atomic_write_creates_the_missing_parent_directories(tmp_path: Path) -> None:
    """Verifies that a destination nested under directories that do not exist has them created."""
    file_path = tmp_path / "nested" / "deeper" / "document.txt"

    with atomic_write(file_path=file_path) as file:
        file.write("payload")

    assert file_path.read_text() == "payload"


def test_atomic_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """Verifies that a completed write leaves the destination alone in its directory."""
    file_path = tmp_path / "document.txt"

    with atomic_write(file_path=file_path) as file:
        file.write("payload")

    assert [entry.name for entry in tmp_path.iterdir()] == ["document.txt"]


def test_atomic_write_discards_the_temporary_file_when_the_body_raises(tmp_path: Path) -> None:
    """Verifies that a failure inside the context leaves the destination untouched and removes the temporary file."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("previous")
    message = "The caller failed partway through writing."

    def failing_write() -> None:
        with atomic_write(file_path=file_path) as file:
            file.write("current")
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        failing_write()

    assert file_path.read_text() == "previous"
    assert [entry.name for entry in tmp_path.iterdir()] == ["document.txt"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows resolves file access through ACLs, not permission bits")
def test_atomic_write_applies_the_default_permissions(tmp_path: Path) -> None:
    """Verifies that a written file carries the permissions the process umask allows."""
    file_path = tmp_path / "document.txt"

    with atomic_write(file_path=file_path) as file:
        file.write("payload")

    # Reads the umask by setting it, since the platform exposes no way to query it without doing so.
    umask = os.umask(0)
    os.umask(umask)

    assert file_path.stat().st_mode & 0o777 == 0o666 & ~umask

    # A group-readable and world-readable file is what a shared acquisition directory depends on, so the common umask
    # is asserted directly against the literal it produces.
    if umask == 0o022:
        assert file_path.stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(sys.platform == "win32", reason="Windows resolves file access through ACLs, not permission bits")
def test_atomic_write_applies_the_default_permissions_to_a_replaced_file(tmp_path: Path) -> None:
    """Verifies that replacing a file carrying narrowed permissions restores the default permissions."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("previous")
    file_path.chmod(0o600)

    with atomic_write(file_path=file_path) as file:
        file.write("current")

    umask = os.umask(0)
    os.umask(umask)

    assert file_path.stat().st_mode & 0o777 == 0o666 & ~umask


def test_direct_write_publishes_text_contents(tmp_path: Path) -> None:
    """Verifies that the written text reaches the destination."""
    file_path = tmp_path / "document.txt"

    with direct_write(file_path=file_path) as file:
        file.write("payload")

    assert file_path.read_text() == "payload"


def test_direct_write_publishes_binary_contents(tmp_path: Path) -> None:
    """Verifies that the binary mode accepts bytes and writes them unchanged."""
    file_path = tmp_path / "document.bin"

    with direct_write(file_path=file_path, binary=True) as file:
        file.write(b"\x00\x01\x02")

    assert file_path.read_bytes() == b"\x00\x01\x02"


def test_direct_write_truncates_an_existing_file(tmp_path: Path) -> None:
    """Verifies that a destination holding longer contents keeps none of them."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("a much longer previous payload")

    with direct_write(file_path=file_path) as file:
        file.write("short")

    assert file_path.read_text() == "short"


def test_direct_write_creates_the_missing_parent_directories(tmp_path: Path) -> None:
    """Verifies that a destination nested under directories that do not exist has them created."""
    file_path = tmp_path / "nested" / "deeper" / "document.txt"

    with direct_write(file_path=file_path) as file:
        file.write("payload")

    assert file_path.read_text() == "payload"


def test_direct_write_uses_no_temporary_file(tmp_path: Path) -> None:
    """Verifies that the contents land on the destination itself rather than through a published sibling."""
    file_path = tmp_path / "document.txt"

    with direct_write(file_path=file_path) as file:
        file.write("payload")

        # The destination carries the path being written, which is what separates this writer from atomic_write().
        assert [entry.name for entry in tmp_path.iterdir()] == ["document.txt"]

    assert file_path.read_text() == "payload"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows resolves file access through ACLs, not permission bits")
def test_direct_write_applies_the_default_permissions(tmp_path: Path) -> None:
    """Verifies that a written file carries the same permissions the atomic writer applies."""
    file_path = tmp_path / "document.txt"

    with direct_write(file_path=file_path) as file:
        file.write("payload")

    umask = os.umask(0)
    os.umask(umask)

    assert file_path.stat().st_mode & 0o777 == 0o666 & ~umask


def test_publish_file_renames_on_posix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that publishing uses a single rename on a non-Windows host."""
    source = tmp_path / "source.tmp"
    source.write_text("payload")
    destination = tmp_path / "destination.txt"

    monkeypatch.setattr(sys, "platform", "linux")

    original_replace = Path.replace
    attempts: list[Path] = []

    def counting_replace(self: Path, target: Path) -> Path:
        attempts.append(target)
        return original_replace(self, target=target)

    monkeypatch.setattr(Path, "replace", counting_replace)

    _publish_file(temporary_path=source, file_path=destination)

    # A single attempt is what separates this branch from the Windows one, which wraps the rename in a retry loop.
    assert len(attempts) == 1
    assert destination.read_text() == "payload"
    assert not source.exists()


def test_publish_file_retries_locked_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that publishing retries the rename while a Windows destination stays locked."""
    source = tmp_path / "source.tmp"
    source.write_text("payload")
    destination = tmp_path / "destination.txt"

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(write_tools, "_RENAME_RETRY_DELAY_MILLISECONDS", 1)

    original_replace = Path.replace
    attempts: list[Path] = []
    message = "The destination is held open by another process."

    def flaky_replace(self: Path, target: Path) -> Path:
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError(message)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    _publish_file(temporary_path=source, file_path=destination)

    assert len(attempts) == 3
    assert destination.read_text() == "payload"
    assert not source.exists()


def test_publish_file_exhausts_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that publishing propagates the failure when a Windows destination never unlocks."""
    source = tmp_path / "source.tmp"
    source.write_text("payload")
    destination = tmp_path / "destination.txt"

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(write_tools, "_RENAME_RETRY_DELAY_MILLISECONDS", 1)

    attempts: list[Path] = []
    message = "The destination is held open by another process."

    def locked_replace(self: Path, target: Path) -> Path:
        attempts.append(target)
        raise PermissionError(message)

    monkeypatch.setattr(Path, "replace", locked_replace)

    with pytest.raises(PermissionError):
        _publish_file(temporary_path=source, file_path=destination)

    assert len(attempts) == _RENAME_RETRY_COUNT
