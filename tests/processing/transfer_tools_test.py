"""Contains tests for the transfer_tools module provided by the processing package."""

import os
import errno
from typing import Any
from pathlib import Path

import pytest
from ataraxis_base_utilities import LogLevel, console

from ataraxis_data_structures import (
    delete_directory,
    transfer_directory,
    calculate_directory_checksum,
)
from ataraxis_data_structures.processing.transfer_tools import _classify_entry


@pytest.fixture
def sample_directory_structure(tmp_path: Path) -> Path:
    """Creates a sample directory structure for testing."""
    root = tmp_path / "test_source"
    root.mkdir()

    (root / "file1.txt").write_text("content1")
    (root / "file2.txt").write_text("content2")

    # Creates subdirectories with files.
    subdirectory_one = root / "subdir1"
    subdirectory_one.mkdir()
    (subdirectory_one / "file3.txt").write_text("content3")
    (subdirectory_one / "file4.txt").write_text("content4")

    subdirectory_two = root / "subdir2"
    subdirectory_two.mkdir()
    (subdirectory_two / "file5.txt").write_text("content5")

    # Creates a nested subdirectory.
    nested = subdirectory_one / "nested"
    nested.mkdir()
    (nested / "file6.txt").write_text("content6")

    return root


@pytest.fixture
def large_directory_structure(tmp_path: Path) -> Path:
    """Creates a larger directory structure for performance testing."""
    root = tmp_path / "large_source"
    root.mkdir()

    # Creates multiple files and subdirectories.
    for file_index in range(20):
        (root / f"file_{file_index}.txt").write_text(f"content_{file_index}" * 100)

    for subdirectory_index in range(5):
        subdirectory = root / f"subdir_{subdirectory_index}"
        subdirectory.mkdir()
        for file_index in range(10):
            (subdirectory / f"file_{file_index}.txt").write_text(
                f"nested_content_{subdirectory_index}_{file_index}" * 50
            )

    return root


def test_delete_directory_basic(tmp_path: Path) -> None:
    """Verifies basic directory deletion functionality."""
    # Creates a simple directory structure.
    test_directory = tmp_path / "to_delete"
    test_directory.mkdir()
    (test_directory / "file1.txt").write_text("content")
    (test_directory / "file2.txt").write_text("content")

    assert test_directory.exists()

    delete_directory(directory_path=test_directory)

    assert not test_directory.exists()


def test_delete_directory_nested(tmp_path: Path) -> None:
    """Verifies deletion of nested directory structures."""
    # Creates nested structure.
    root = tmp_path / "nested_root"
    root.mkdir()
    level1 = root / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()

    # Adds files at each level.
    (root / "file1.txt").write_text("content1")
    (level1 / "file2.txt").write_text("content2")
    (level2 / "file3.txt").write_text("content3")

    # Deletes entire structure.
    delete_directory(directory_path=root)

    # Verifies all levels are deleted.
    assert not root.exists()
    assert not level1.exists()
    assert not level2.exists()


def test_delete_directory_nonexistent(tmp_path: Path) -> None:
    """Verifies that deleting a non-existent directory does not raise errors."""
    nonexistent = tmp_path / "does_not_exist"
    # Completes without raising an exception.
    delete_directory(directory_path=nonexistent)


def test_delete_directory_empty(tmp_path: Path) -> None:
    """Verifies deletion of empty directories."""
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()

    assert empty_directory.exists()
    delete_directory(directory_path=empty_directory)
    assert not empty_directory.exists()


def test_transfer_directory_basic(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies basic directory transfer functionality."""
    source = sample_directory_structure
    destination = tmp_path / "test_destination"

    transfer_directory(source=source, destination=destination)

    assert destination.exists()

    # Verifies all files were transferred.
    assert (destination / "file1.txt").exists()
    assert (destination / "file2.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()
    assert (destination / "subdir1" / "file4.txt").exists()
    assert (destination / "subdir2" / "file5.txt").exists()
    assert (destination / "subdir1" / "nested" / "file6.txt").exists()

    # Verifies content integrity.
    assert (destination / "file1.txt").read_text() == "content1"
    assert (destination / "subdir1" / "nested" / "file6.txt").read_text() == "content6"

    # Verifies source still exists (no removal).
    assert source.exists()


@pytest.mark.parametrize("num_threads", [1, 2, 4, -1])
def test_transfer_directory_multithreading(sample_directory_structure: Path, tmp_path: Path, num_threads: int) -> None:
    """Verifies that transfer_directory works correctly with different thread counts."""
    source = sample_directory_structure
    destination = tmp_path / f"dest_threads_{num_threads}"

    transfer_directory(source=source, destination=destination, num_threads=num_threads)

    # Verifies all files were transferred correctly.
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()
    assert (destination / "subdir1" / "nested" / "file6.txt").exists()

    # Verifies content.
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_with_removal(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that the source directory is removed when remove_source=True."""
    source = sample_directory_structure
    destination = tmp_path / "dest_with_removal"

    # Stores original file count.
    original_files = list(source.rglob("*.txt"))
    assert original_files

    transfer_directory(source=source, destination=destination, remove_source=True)

    transferred_files = list(destination.rglob("*.txt"))
    assert len(transferred_files) == len(original_files)

    assert not source.exists()


def test_transfer_directory_with_integrity_check(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies the integrity verification feature of transfer_directory."""
    source = sample_directory_structure
    destination = tmp_path / "dest_integrity"

    transfer_directory(source=source, destination=destination, verify_integrity=True)

    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    # Verifies the checksum file was created in the source.
    assert (source / "ax_checksum.txt").exists()


def test_transfer_directory_with_existing_checksum(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies transfer when the checksum file already exists."""
    source = sample_directory_structure
    destination = tmp_path / "dest_existing_checksum"

    # Pre-creates checksum.
    calculate_directory_checksum(directory=source, progress=False, save_checksum=True)
    assert (source / "ax_checksum.txt").exists()

    transfer_directory(source=source, destination=destination, verify_integrity=True)

    assert destination.exists()
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_nonexistent_source(tmp_path: Path) -> None:
    """Verifies that transferring a non-existent source raises FileNotFoundError."""
    source = tmp_path / "nonexistent"
    destination = tmp_path / "destination"

    with pytest.raises(FileNotFoundError):
        transfer_directory(source=source, destination=destination)


def test_transfer_directory_preserves_structure(tmp_path: Path) -> None:
    """Verifies that complex directory hierarchies are preserved during transfer."""
    # Creates complex structure.
    source = tmp_path / "complex_source"
    source.mkdir()

    # Creates multiple levels.
    (source / "level1").mkdir()
    (source / "level1" / "level2").mkdir()
    (source / "level1" / "level2" / "level3").mkdir()
    (source / "level1" / "sibling").mkdir()

    # Adds files at different levels.
    (source / "root.txt").write_text("root")
    (source / "level1" / "l1.txt").write_text("level1")
    (source / "level1" / "level2" / "l2.txt").write_text("level2")
    (source / "level1" / "level2" / "level3" / "l3.txt").write_text("level3")
    (source / "level1" / "sibling" / "sib.txt").write_text("sibling")

    destination = tmp_path / "complex_dest"
    transfer_directory(source=source, destination=destination)

    assert (destination / "root.txt").exists()
    assert (destination / "level1" / "l1.txt").exists()
    assert (destination / "level1" / "level2" / "l2.txt").exists()
    assert (destination / "level1" / "level2" / "level3" / "l3.txt").exists()
    assert (destination / "level1" / "sibling" / "sib.txt").exists()

    # Verifies content.
    assert (destination / "level1" / "level2" / "level3" / "l3.txt").read_text() == "level3"


def test_transfer_directory_empty_source(tmp_path: Path) -> None:
    """Verifies transfer of an empty directory."""
    source = tmp_path / "empty_source"
    source.mkdir()
    destination = tmp_path / "empty_dest"

    transfer_directory(source=source, destination=destination)

    # Verifies destination exists but is empty.
    assert destination.exists()
    assert not list(destination.iterdir())


def test_transfer_directory_large_dataset(large_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies transfer of a larger directory structure with multiple threads."""
    source = large_directory_structure
    destination = tmp_path / "large_dest"

    # Counts files in the source.
    source_files = list(source.rglob("*.txt"))
    source_count = len(source_files)

    transfer_directory(source=source, destination=destination, num_threads=4)

    destination_files = list(destination.rglob("*.txt"))
    assert len(destination_files) == source_count

    # Spot checks some files.
    assert (destination / "file_0.txt").exists()
    assert (destination / "subdir_0" / "file_0.txt").exists()


def test_transfer_directory_with_integrity_and_removal(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies combined integrity verification and source removal."""
    source = sample_directory_structure
    destination = tmp_path / "dest_integrity_removal"

    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
        remove_source=True,
    )

    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    assert not source.exists()


def test_delete_directory_parallel_performance(tmp_path: Path) -> None:
    """Verifies that parallel deletion works with many files."""
    # Creates a directory with many files.
    test_directory = tmp_path / "many_files"
    test_directory.mkdir()

    for file_index in range(100):
        (test_directory / f"file_{file_index}.txt").write_text(f"content_{file_index}")

    # Creates subdirectories.
    for subdirectory_index in range(10):
        subdirectory = test_directory / f"subdir_{subdirectory_index}"
        subdirectory.mkdir()
        for file_index in range(10):
            (subdirectory / f"file_{file_index}.txt").write_text(f"content_{subdirectory_index}_{file_index}")

    assert test_directory.exists()
    file_count = len(list(test_directory.rglob("*.txt")))
    assert file_count == 200

    delete_directory(directory_path=test_directory)

    assert not test_directory.exists()


def test_transfer_directory_metadata_preservation(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that file metadata is preserved during transfer."""
    source = sample_directory_structure
    destination = tmp_path / "dest_metadata"

    # Gets original file stats.
    original_file = source / "file1.txt"
    original_stat = original_file.stat()

    transfer_directory(source=source, destination=destination)

    # Gets transferred file stats.
    transferred_file = destination / "file1.txt"
    transferred_stat = transferred_file.stat()

    # Verifies metadata (shutil.copy2 should preserve modification time).
    assert transferred_stat.st_size == original_stat.st_size
    # Filesystems preserve the modification time only approximately, so the comparison allows a one second delta.
    assert abs(transferred_stat.st_mtime - original_stat.st_mtime) < 1


def test_transfer_directory_to_existing_destination(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies transfer into a destination that already exists and holds nothing the source does not account for."""
    source = sample_directory_structure
    destination = tmp_path / "existing_dest"

    # Pre-creates the destination, along with a subdirectory the source also provides.
    destination.mkdir()
    (destination / "subdir1").mkdir()

    transfer_directory(source=source, destination=destination)

    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()


def test_transfer_directory_creates_a_dotted_destination(tmp_path: Path) -> None:
    """Verifies that a destination whose own name carries a dot is created as a directory rather than skipped.

    The source deliberately holds no subdirectory. Recreating the source hierarchy calls mkdir() with 'parents' set,
    so a source that carries even one subdirectory creates the destination as a side effect and hides the defect. With
    a flat source, the directory check is the only step that can create the destination, and leaving it to the suffix
    heuristic creates the parent alone and fails the copy against a destination that does not exist.
    """
    source = tmp_path / "flat_source"
    source.mkdir()
    (source / "file1.txt").write_text("content1")
    (source / "file2.txt").write_text("content2")
    destination = tmp_path / "session_2026.08.05"

    transfer_directory(source=source, destination=destination)

    assert destination.is_dir()
    assert (destination / "file1.txt").read_text() == "content1"
    assert (destination / "file2.txt").read_text() == "content2"


def test_transfer_directory_rejects_a_dirty_destination(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that a destination holding unaccounted files is rejected rather than failing the integrity check."""
    source = sample_directory_structure
    destination = tmp_path / "dirty_dest"
    destination.mkdir()
    (destination / "stray.txt").write_text("left over from an earlier transfer")

    with pytest.raises(RuntimeError, match="does not account for"):
        transfer_directory(source=source, destination=destination)

    # The rejected transfer left the destination exactly as it found it.
    assert (destination / "stray.txt").exists()
    assert not (destination / "file1.txt").exists()


def test_transfer_directory_resets_a_dirty_destination(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that the reset flag deletes the unaccounted files and lets the verified transfer proceed."""
    source = sample_directory_structure
    destination = tmp_path / "reset_dest"
    (destination / "subdir1").mkdir(parents=True)
    (destination / "stray.txt").write_text("left over from an earlier transfer")
    (destination / "subdir1" / "nested_stray.txt").write_text("also left over")

    transfer_directory(source=source, destination=destination, verify_integrity=True, reset_dirty_destination=True)

    # Only the unaccounted files were removed, and the transfer verified cleanly against the cleaned destination.
    assert not (destination / "stray.txt").exists()
    assert not (destination / "subdir1" / "nested_stray.txt").exists()
    assert (destination / "file1.txt").read_text() == "content1"
    assert (destination / "subdir1" / "file3.txt").read_text() == "content3"


def test_transfer_directory_rejects_a_source_holding_symlinks(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that a source tree containing a symlink is refused before anything is written."""
    source = sample_directory_structure
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.txt").write_text("data the link stands for")
    (source / "linked").symlink_to(outside, target_is_directory=True)

    destination = tmp_path / "symlink_dest"

    with pytest.raises(RuntimeError, match="Resolve the following link"):
        transfer_directory(source=source, destination=destination, verify_integrity=True)

    # The refusal precedes every side effect, so no checksum was written and no destination entry was created.
    assert not (source / "ax_checksum.txt").exists()
    assert not destination.exists()
    assert (outside / "payload.txt").exists()


def test_transfer_directory_single_vs_multi_thread_consistency(
    sample_directory_structure: Path, tmp_path: Path
) -> None:
    """Verifies that single-threaded and multithreaded transfers produce identical results."""
    source = sample_directory_structure
    destination_single = tmp_path / "dest_single"
    destination_multi = tmp_path / "dest_multi"

    # Single-threaded transfer.
    transfer_directory(source=source, destination=destination_single, num_threads=1)

    # Multithreaded transfer.
    transfer_directory(source=source, destination=destination_multi, num_threads=4)

    # Compares file lists.
    single_files = sorted(
        file_path.relative_to(destination_single) for file_path in destination_single.rglob("*") if file_path.is_file()
    )
    multi_files = sorted(
        file_path.relative_to(destination_multi) for file_path in destination_multi.rglob("*") if file_path.is_file()
    )

    assert single_files == multi_files

    # Verifies content matches.
    for relative_path in single_files:
        single_content = (destination_single / relative_path).read_text()
        multi_content = (destination_multi / relative_path).read_text()
        assert single_content == multi_content


def test_transfer_directory_integrity_check_detects_corruption(
    sample_directory_structure: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that integrity verification detects corrupted transfers."""
    source = sample_directory_structure
    destination = tmp_path / "dest_corrupted"

    # Tracks which directory is being checksummed.
    checksum_calls = []
    original_calculate_checksum = calculate_directory_checksum

    def mock_calculate_checksum(directory: Path, **kwargs: Any) -> str:
        """Mocks calculate_directory_checksum to return different values for source and destination."""
        checksum_calls.append(directory)
        result = original_calculate_checksum(directory=directory, **kwargs)

        # Returns different checksum for destination to simulate corruption.
        if directory == destination:
            return "corrupted_checksum_00000000000000"
        return result

    monkeypatch.setattr(
        target="ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum",
        name=mock_calculate_checksum,
    )

    # Attempts transfer with integrity verification.
    with pytest.raises(RuntimeError) as exception_info:
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
        )

    # Verifies the error message contains expected information.
    # Normalizes whitespace since the error message may contain line breaks.
    error_message = str(exception_info.value).replace("\n", " ")
    assert "Unable to verify the integrity of the directory transferred" in error_message
    assert "corrupted in transmission" in error_message

    # Verifies both source and destination were checksummed.
    assert len(checksum_calls) >= 2  # At least initial checksum and verification.


def test_transfer_directory_checksum_path_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that error messages truncate long paths to the last 6 parts."""
    # Creates the deeply nested source path.
    source = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "source"
    source.mkdir(parents=True)
    (source / "file.txt").write_text("content")

    destination = tmp_path / "x" / "y" / "z" / "w" / "v" / "u" / "dest"
    destination.mkdir(parents=True)

    # Pre-calculates checksum.
    calculate_directory_checksum(directory=source, save_checksum=True)

    # Tracks the original function.
    original_calculate_checksum = calculate_directory_checksum

    # Mocks calculate_directory_checksum to return a corrupted hash for destination.
    def mock_calculate_checksum(directory: Path, **kwargs: Any) -> str:
        """Mocks calculate_directory_checksum to return a corrupted checksum for the destination directory."""
        result = original_calculate_checksum(directory=directory, **kwargs)
        if directory == destination:
            return "corrupted_hash_00000000000000"
        return result

    monkeypatch.setattr(
        target="ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum",
        name=mock_calculate_checksum,
    )

    with pytest.raises(RuntimeError) as exception_info:
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
        )

    # Verifies the error message contains truncated paths.
    error_message = str(exception_info.value)
    assert "Unable to verify the integrity of the directory transferred" in error_message

    # Verifies the rendered paths keep the deepest components (the last 6 are b/c/d/e/f/source and y/z/w/v/u/dest).
    assert "e/f/source" in error_message or "e\\f\\source" in error_message  # Unix or Windows path separator.
    assert "v/u/dest" in error_message or "v\\u\\dest" in error_message


def test_transfer_directory_integrity_check_corruption_prevents_removal(
    sample_directory_structure: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the source is NOT removed when an integrity check fails."""
    source = sample_directory_structure
    destination = tmp_path / "dest_corrupted_no_removal"

    # Mocks checksum to simulate corruption.
    original_calculate_checksum = calculate_directory_checksum

    def mock_calculate_checksum(directory: Path, **kwargs: Any) -> str:
        """Mocks calculate_directory_checksum to return a corrupted checksum for the destination directory."""
        result = original_calculate_checksum(directory=directory, **kwargs)
        if directory == destination:
            return "different_checksum_1234567890abcd"
        return result

    monkeypatch.setattr(
        target="ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum",
        name=mock_calculate_checksum,
    )

    # Attempts transfer with both verification and removal enabled.
    with pytest.raises(RuntimeError):
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
            remove_source=True,
        )

    # Verifies the source still exists (was not removed due to failed verification).
    assert source.exists()
    assert (source / "file1.txt").exists()


def test_transfer_directory_integrity_check_with_progress(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that integrity verification works with progress tracking enabled."""
    source = sample_directory_structure
    destination = tmp_path / "dest_progress_integrity"

    # Performs transfer with both progress and integrity enabled.
    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
        progress=True,
    )

    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    # Verifies the checksum file exists.
    assert (source / "ax_checksum.txt").exists()


def test_transfer_directory_multithreaded_with_progress(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that a multithreaded transfer reports progress while copying every file."""
    destination = tmp_path / "dest_multithreaded_progress"

    # The progress bar wraps the completion of the submitted futures, which is a separate loop from the one the
    # single-threaded transfer tracks, so reaching it takes both a thread count above one and progress enabled.
    transfer_directory(
        source=sample_directory_structure,
        destination=destination,
        num_threads=4,
        progress=True,
    )

    assert (destination / "file1.txt").read_text() == "content1"
    assert (destination / "subdir1" / "file3.txt").read_text() == "content3"
    assert (destination / "subdir1" / "nested" / "file6.txt").read_text() == "content6"


def test_delete_directory_retries_a_refused_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that directory removal retries after an attempt that the filesystem refuses."""
    target = tmp_path / "retried"
    target.mkdir()
    (target / "data.txt").write_text("payload")

    real_rmdir = Path.rmdir
    attempts: list[str] = []

    def _refuse_first_attempt(self: Path) -> None:
        # Windows refuses the removal while a handle to a just-unlinked entry is still open, which is the transient
        # failure the retry loop absorbs.
        attempts.append(str(self))
        if len(attempts) == 1:
            raise PermissionError(errno.EACCES, "Injected failure", str(self))
        real_rmdir(self)

    monkeypatch.setattr(target=Path, name="rmdir", value=_refuse_first_attempt)

    delete_directory(directory_path=target)

    assert len(attempts) > 1
    assert not target.exists()


def test_delete_directory_warns_when_every_removal_attempt_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that directory removal reports a warning when it exhausts every attempt."""
    target = tmp_path / "undeletable"
    target.mkdir()
    (target / "data.txt").write_text("payload")

    def _refuse_every_attempt(self: Path) -> None:
        raise PermissionError(errno.EACCES, "Injected failure", str(self))

    monkeypatch.setattr(target=Path, name="rmdir", value=_refuse_every_attempt)

    # Records the console call directly, since loguru writes through a stream reference that the pytest capture
    # fixtures do not intercept.
    reported: list[tuple[str, str]] = []
    monkeypatch.setattr(console, "echo", lambda message, level: reported.append((message, level)))

    delete_directory(directory_path=target)

    assert len(reported) == 1
    reported_message, reported_level = reported[0]
    assert reported_level == LogLevel.WARNING
    assert f"Unable to remove the {target} directory after 5 attempts" in reported_message

    # The warning stands in for the removal, so the directory survives while its contents are already unlinked.
    assert target.exists()


def test_transfer_directory_creates_checksum_when_missing(tmp_path: Path) -> None:
    """Verifies that checksum is automatically created if missing when verify_integrity=True."""
    # Creates the source without a pre-calculated checksum.
    source = tmp_path / "source_no_checksum"
    source.mkdir()
    (source / "file1.txt").write_text("content1")
    (source / "file2.txt").write_text("content2")

    destination = tmp_path / "dest_auto_checksum"

    # Verifies no checksum exists initially.
    assert not (source / "ax_checksum.txt").exists()

    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
    )

    # Verifies checksum was automatically created.
    assert (source / "ax_checksum.txt").exists()

    assert destination.exists()
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_preserves_checksum_file(sample_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that the original checksum file is preserved in the source."""
    source = sample_directory_structure
    destination = tmp_path / "dest_checksum_preserved"

    # Verifies no checksum initially.
    assert not (source / "ax_checksum.txt").exists()

    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
    )

    # Verifies the checksum file exists and persists in the source.
    assert (source / "ax_checksum.txt").exists()

    # Verifies the checksum file is readable and valid.
    checksum_content = (source / "ax_checksum.txt").read_text().strip()
    assert len(checksum_content) == 32  # xxHash3-128 hex string.
    assert all(character in "0123456789abcdef" for character in checksum_content)


def test_transfer_directory_integrity_multithread_consistency(large_directory_structure: Path, tmp_path: Path) -> None:
    """Verifies that integrity checking works correctly with multithreaded transfers."""
    source = large_directory_structure
    destination = tmp_path / "dest_multi_integrity"

    transfer_directory(
        source=source,
        destination=destination,
        num_threads=4,
        verify_integrity=True,
    )

    # Verifies all files transferred correctly.
    source_files = sorted(file_path.relative_to(source) for file_path in source.rglob("*.txt"))
    destination_files = sorted(file_path.relative_to(destination) for file_path in destination.rglob("*.txt"))

    assert source_files == destination_files

    # Spot checks file contents.
    assert (destination / "file_0.txt").exists()
    assert (destination / "subdir_0" / "file_0.txt").exists()

    # Verifies the source checksum file was created.
    assert (source / "ax_checksum.txt").exists()


def test_delete_directory_unlinks_a_directory_symlink_without_following_it(tmp_path: Path) -> None:
    """Verifies that a symlinked subdirectory is removed as a link, leaving the tree it points at untouched."""
    target = tmp_path / "to_delete"
    target.mkdir()
    (target / "own.txt").write_text("disposable")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me")
    (outside / "nested").mkdir()
    (outside / "nested" / "also_precious.txt").write_text("keep me too")

    (target / "link").symlink_to(outside, target_is_directory=True)

    delete_directory(directory_path=target)

    # The named tree is gone, and nothing behind the link was touched.
    assert not target.exists()
    assert (outside / "precious.txt").read_text() == "keep me"
    assert (outside / "nested" / "also_precious.txt").read_text() == "keep me too"


def test_delete_directory_removes_entries_that_are_neither_files_nor_directories(tmp_path: Path) -> None:
    """Verifies that a dangling symlink is removed, since it reports as neither a file nor a directory."""
    root = tmp_path / "with_dangling_link"
    root.mkdir()
    (root / "regular.txt").write_text("content")
    (root / "dangling").symlink_to(root / "missing_target")

    delete_directory(directory_path=root)

    assert not root.exists()


def test_classify_entry_reports_each_entry_kind(tmp_path: Path) -> None:
    """Verifies that _classify_entry separates symlinks, directories, and files by their link-level metadata."""
    (tmp_path / "regular.txt").write_text("content")
    (tmp_path / "subdirectory").mkdir()
    (tmp_path / "link_to_file").symlink_to(tmp_path / "regular.txt")
    (tmp_path / "link_to_directory").symlink_to(tmp_path / "subdirectory", target_is_directory=True)

    assert _classify_entry(path=tmp_path / "regular.txt") == (False, False)
    assert _classify_entry(path=tmp_path / "subdirectory") == (False, True)

    # A link answers the directory question with False whatever it points at.
    assert _classify_entry(path=tmp_path / "link_to_file") == (True, False)
    assert _classify_entry(path=tmp_path / "link_to_directory") == (True, False)


def test_classify_entry_reports_a_vanished_entry_as_a_plain_file(tmp_path: Path) -> None:
    """Verifies that _classify_entry answers False to both questions when the entry no longer exists."""
    assert _classify_entry(path=tmp_path / "never_created.bin") == (False, False)


def test_classify_entry_propagates_an_unreadable_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _classify_entry raises when the entry exists but its metadata cannot be read."""
    target = tmp_path / "unreadable.bin"
    target.write_text("content")

    def _deny_metadata(_path: Path) -> os.stat_result:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(target=os, name="lstat", value=_deny_metadata)

    # A propagated failure is what keeps an unreadable symlink from being filed as a plain file, which would carry it
    # past the link rejection transfer_directory performs.
    with pytest.raises(PermissionError):
        _classify_entry(path=target)
