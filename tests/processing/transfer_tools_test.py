"""Contains tests for the transfer_tools module provided by the processing package."""

from pathlib import Path
import multiprocessing

import pytest

from ataraxis_data_structures import (
    delete_directory,
    transfer_directory,
    calculate_directory_checksum,
)

multiprocessing.set_start_method("spawn", force=True)  # Improves reproducibility.


@pytest.fixture
def sample_directory_structure(tmp_path) -> Path:
    """Creates a sample directory structure for testing."""
    root = tmp_path / "test_source"
    root.mkdir()

    # Creates files in root
    (root / "file1.txt").write_text("content1")
    (root / "file2.txt").write_text("content2")

    # Creates subdirectories with files
    subdir1 = root / "subdir1"
    subdir1.mkdir()
    (subdir1 / "file3.txt").write_text("content3")
    (subdir1 / "file4.txt").write_text("content4")

    subdir2 = root / "subdir2"
    subdir2.mkdir()
    (subdir2 / "file5.txt").write_text("content5")

    # Creates a nested subdirectory
    nested = subdir1 / "nested"
    nested.mkdir()
    (nested / "file6.txt").write_text("content6")

    return root


@pytest.fixture
def large_directory_structure(tmp_path) -> Path:
    """Creates a larger directory structure for performance testing."""
    root = tmp_path / "large_source"
    root.mkdir()

    # Creates multiple files and subdirectories
    for i in range(20):
        (root / f"file_{i}.txt").write_text(f"content_{i}" * 100)

    for i in range(5):
        subdir = root / f"subdir_{i}"
        subdir.mkdir()
        for j in range(10):
            (subdir / f"file_{j}.txt").write_text(f"nested_content_{i}_{j}" * 50)

    return root


def test_delete_directory_basic(tmp_path):
    """Verifies basic directory deletion functionality."""
    # Creates a simple directory structure
    test_dir = tmp_path / "to_delete"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content")
    (test_dir / "file2.txt").write_text("content")

    # Verifies directory exists
    assert test_dir.exists()

    # Deletes directory
    delete_directory(test_dir)

    # Verifies directory is gone
    assert not test_dir.exists()


def test_delete_directory_nested(tmp_path):
    """Verifies deletion of nested directory structures."""
    # Creates nested structure
    root = tmp_path / "nested_root"
    root.mkdir()
    level1 = root / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()

    # Adds files at each level
    (root / "file1.txt").write_text("content1")
    (level1 / "file2.txt").write_text("content2")
    (level2 / "file3.txt").write_text("content3")

    # Deletes entire structure
    delete_directory(root)

    # Verifies all levels are deleted
    assert not root.exists()
    assert not level1.exists()
    assert not level2.exists()


def test_delete_directory_nonexistent(tmp_path):
    """Verifies that deleting a non-existent directory does not raise errors."""
    nonexistent = tmp_path / "does_not_exist"
    # Should not raise any exception
    delete_directory(nonexistent)


def test_delete_directory_empty(tmp_path):
    """Verifies deletion of empty directories."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert empty_dir.exists()
    delete_directory(empty_dir)
    assert not empty_dir.exists()


def test_transfer_directory_basic(sample_directory_structure, tmp_path):
    """Verifies basic directory transfer functionality."""
    source = sample_directory_structure
    destination = tmp_path / "test_destination"

    # Performs transfer
    transfer_directory(source=source, destination=destination)

    # Verifies destination exists
    assert destination.exists()

    # Verifies all files were transferred
    assert (destination / "file1.txt").exists()
    assert (destination / "file2.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()
    assert (destination / "subdir1" / "file4.txt").exists()
    assert (destination / "subdir2" / "file5.txt").exists()
    assert (destination / "subdir1" / "nested" / "file6.txt").exists()

    # Verifies content integrity
    assert (destination / "file1.txt").read_text() == "content1"
    assert (destination / "subdir1" / "nested" / "file6.txt").read_text() == "content6"

    # Verifies source still exists (no removal)
    assert source.exists()


@pytest.mark.parametrize("num_threads", [1, 2, 4, -1])
def test_transfer_directory_multithreading(sample_directory_structure, tmp_path, num_threads):
    """Verifies that transfer_directory works correctly with different thread counts."""
    source = sample_directory_structure
    destination = tmp_path / f"dest_threads_{num_threads}"

    transfer_directory(source=source, destination=destination, num_threads=num_threads)

    # Verifies all files were transferred correctly
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()
    assert (destination / "subdir1" / "nested" / "file6.txt").exists()

    # Verifies content
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_with_removal(sample_directory_structure, tmp_path):
    """Verifies that the source directory is removed when remove_source=True."""
    source = sample_directory_structure
    destination = tmp_path / "dest_with_removal"

    # Stores original file count
    original_files = list(source.rglob("*.txt"))
    assert len(original_files) > 0

    # Performs transfer with removal
    transfer_directory(source=source, destination=destination, remove_source=True)

    # Verifies destination has all files
    transferred_files = list(destination.rglob("*.txt"))
    assert len(transferred_files) == len(original_files)

    # Verifies the source is deleted
    assert not source.exists()


def test_transfer_directory_with_integrity_check(sample_directory_structure, tmp_path):
    """Verifies the integrity verification feature of transfer_directory."""
    source = sample_directory_structure
    destination = tmp_path / "dest_integrity"

    # Performs transfer with integrity verification
    transfer_directory(source=source, destination=destination, verify_integrity=True)

    # Verifies destination exists and has correct files
    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    # Verifies the checksum file was created in the source
    assert (source / "ax_checksum.txt").exists()


def test_transfer_directory_with_existing_checksum(sample_directory_structure, tmp_path):
    """Verifies transfer when the checksum file already exists."""
    source = sample_directory_structure
    destination = tmp_path / "dest_existing_checksum"

    # Pre-creates checksum
    calculate_directory_checksum(directory=source, progress=False, save_checksum=True)
    assert (source / "ax_checksum.txt").exists()

    # Performs transfer with verification
    transfer_directory(source=source, destination=destination, verify_integrity=True)

    # Verifies successful transfer
    assert destination.exists()
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_nonexistent_source(tmp_path):
    """Verifies that transferring a non-existent source raises FileNotFoundError."""
    source = tmp_path / "nonexistent"
    destination = tmp_path / "destination"

    with pytest.raises(FileNotFoundError):
        transfer_directory(source=source, destination=destination)


def test_transfer_directory_preserves_structure(tmp_path):
    """Verifies that complex directory hierarchies are preserved during transfer."""
    # Creates complex structure
    source = tmp_path / "complex_source"
    source.mkdir()

    # Creates multiple levels
    (source / "level1").mkdir()
    (source / "level1" / "level2").mkdir()
    (source / "level1" / "level2" / "level3").mkdir()
    (source / "level1" / "sibling").mkdir()

    # Adds files at different levels
    (source / "root.txt").write_text("root")
    (source / "level1" / "l1.txt").write_text("level1")
    (source / "level1" / "level2" / "l2.txt").write_text("level2")
    (source / "level1" / "level2" / "level3" / "l3.txt").write_text("level3")
    (source / "level1" / "sibling" / "sib.txt").write_text("sibling")

    destination = tmp_path / "complex_dest"
    transfer_directory(source=source, destination=destination)

    # Verifies structure
    assert (destination / "root.txt").exists()
    assert (destination / "level1" / "l1.txt").exists()
    assert (destination / "level1" / "level2" / "l2.txt").exists()
    assert (destination / "level1" / "level2" / "level3" / "l3.txt").exists()
    assert (destination / "level1" / "sibling" / "sib.txt").exists()

    # Verifies content
    assert (destination / "level1" / "level2" / "level3" / "l3.txt").read_text() == "level3"


def test_transfer_directory_empty_source(tmp_path):
    """Verifies transfer of an empty directory."""
    source = tmp_path / "empty_source"
    source.mkdir()
    destination = tmp_path / "empty_dest"

    transfer_directory(source=source, destination=destination)

    # Verifies destination exists but is empty
    assert destination.exists()
    assert len(list(destination.iterdir())) == 0


def test_transfer_directory_large_dataset(large_directory_structure, tmp_path):
    """Verifies transfer of a larger directory structure with multiple threads."""
    source = large_directory_structure
    destination = tmp_path / "large_dest"

    # Counts files in the source
    source_files = list(source.rglob("*.txt"))
    source_count = len(source_files)

    # Performs parallel transfer
    transfer_directory(source=source, destination=destination, num_threads=4)

    # Verifies all files transferred
    dest_files = list(destination.rglob("*.txt"))
    assert len(dest_files) == source_count

    # Spot checks some files
    assert (destination / "file_0.txt").exists()
    assert (destination / "subdir_0" / "file_0.txt").exists()


def test_transfer_directory_with_integrity_and_removal(sample_directory_structure, tmp_path):
    """Verifies combined integrity verification and source removal."""
    source = sample_directory_structure
    destination = tmp_path / "dest_integrity_removal"

    # Performs transfer with both options
    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
        remove_source=True,
    )

    # Verifies destination has files
    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    # Verifies the source is removed
    assert not source.exists()


def test_delete_directory_parallel_performance(tmp_path):
    """Verifies that parallel deletion works with many files."""
    # Creates a directory with many files
    test_dir = tmp_path / "many_files"
    test_dir.mkdir()

    for i in range(100):
        (test_dir / f"file_{i}.txt").write_text(f"content_{i}")

    # Creates subdirectories
    for i in range(10):
        subdir = test_dir / f"subdir_{i}"
        subdir.mkdir()
        for j in range(10):
            (subdir / f"file_{j}.txt").write_text(f"content_{i}_{j}")

    # Verifies creation
    assert test_dir.exists()
    file_count = len(list(test_dir.rglob("*.txt")))
    assert file_count == 200

    # Deletes in parallel
    delete_directory(test_dir)

    # Verifies deletion
    assert not test_dir.exists()


def test_transfer_directory_metadata_preservation(sample_directory_structure, tmp_path):
    """Verifies that file metadata is preserved during transfer."""
    source = sample_directory_structure
    destination = tmp_path / "dest_metadata"

    # Gets original file stats
    original_file = source / "file1.txt"
    original_stat = original_file.stat()

    # Performs transfer
    transfer_directory(source=source, destination=destination)

    # Gets transferred file stats
    transferred_file = destination / "file1.txt"
    transferred_stat = transferred_file.stat()

    # Verifies metadata (shutil.copy2 should preserve modification time)
    assert transferred_stat.st_size == original_stat.st_size
    # Note: Depending on filesystem, mtime might not be exactly preserved
    # but should be very close
    assert abs(transferred_stat.st_mtime - original_stat.st_mtime) < 1


def test_transfer_directory_to_existing_destination(sample_directory_structure, tmp_path):
    """Verifies transfer when the destination directory already exists."""
    source = sample_directory_structure
    destination = tmp_path / "existing_dest"

    # Pre-creates destination
    destination.mkdir()
    (destination / "existing_file.txt").write_text("existing")

    # Performs transfer
    transfer_directory(source=source, destination=destination)

    # Verifies both old and new files exist
    assert (destination / "existing_file.txt").exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()


def test_transfer_directory_single_vs_multi_thread_consistency(sample_directory_structure, tmp_path):
    """Verifies that single-threaded and multithreaded transfers produce identical results."""
    source = sample_directory_structure
    dest_single = tmp_path / "dest_single"
    dest_multi = tmp_path / "dest_multi"

    # Single-threaded transfer
    transfer_directory(source=source, destination=dest_single, num_threads=1)

    # Multithreaded transfer
    transfer_directory(source=source, destination=dest_multi, num_threads=4)

    # Compares file lists
    single_files = sorted([f.relative_to(dest_single) for f in dest_single.rglob("*") if f.is_file()])
    multi_files = sorted([f.relative_to(dest_multi) for f in dest_multi.rglob("*") if f.is_file()])

    assert single_files == multi_files

    # Verifies content matches
    for rel_path in single_files:
        single_content = (dest_single / rel_path).read_text()
        multi_content = (dest_multi / rel_path).read_text()
        assert single_content == multi_content


def test_transfer_directory_integrity_check_detects_corruption(sample_directory_structure, tmp_path, monkeypatch):
    """Verifies that integrity verification detects corrupted transfers."""
    source = sample_directory_structure
    destination = tmp_path / "dest_corrupted"

    # Tracks which directory is being checksummed
    checksum_calls = []
    original_calculate_checksum = calculate_directory_checksum

    def mock_calculate_checksum(directory, **kwargs):
        """Mocks calculate_directory_checksum to return different values for source and destination."""
        checksum_calls.append(directory)
        result = original_calculate_checksum(directory=directory, **kwargs)

        # Returns different checksum for destination to simulate corruption
        if directory == destination:
            return "corrupted_checksum_00000000000000"
        return result

    # Applies monkeypatch
    monkeypatch.setattr(
        "ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum", mock_calculate_checksum
    )

    # Attempts transfer with integrity verification
    with pytest.raises(RuntimeError) as exc_info:
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
        )

    # Verifies the error message contains expected information
    # Normalizes whitespace since the error message may contain line breaks
    error_message = str(exc_info.value).replace("\n", " ")
    assert "Checksum mismatch detected" in error_message
    assert "corrupted in transmission" in error_message

    # Verifies both source and destination were checksummed
    assert len(checksum_calls) >= 2  # At least initial checksum and verification


def test_transfer_directory_checksum_path_truncation(tmp_path, monkeypatch):
    """Verifies that error messages truncate long paths to the last 6 parts."""
    # Creates the deeply nested source path
    source = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "source"
    source.mkdir(parents=True)
    (source / "file.txt").write_text("content")

    destination = tmp_path / "x" / "y" / "z" / "w" / "v" / "u" / "dest"
    destination.mkdir(parents=True)

    # Pre-calculates checksum
    calculate_directory_checksum(directory=source, save_checksum=True)

    # Tracks the original function
    original_calculate_checksum = calculate_directory_checksum

    # Mocks calculate_directory_checksum to return a corrupted hash for destination
    def mock_calculate(directory, **kwargs):
        result = original_calculate_checksum(directory=directory, **kwargs)
        if directory == destination:
            return "corrupted_hash_00000000000000"
        return result

    # Applies a monkeypatch to where the function is called in transfer_tools
    monkeypatch.setattr(
        "ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum", mock_calculate
    )

    with pytest.raises(RuntimeError) as exc_info:
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
        )

    # Verifies the error message contains truncated paths (not full paths)
    error_message = str(exc_info.value)
    assert "Checksum mismatch detected" in error_message

    # Verifies the paths show the last 6 parts (e/f/source and v/u/dest)
    assert "e/f/source" in error_message or "e\\f\\source" in error_message  # Unix or Windows path separator
    assert "v/u/dest" in error_message or "v\\u\\dest" in error_message


def test_transfer_directory_integrity_check_corruption_prevents_removal(
    sample_directory_structure, tmp_path, monkeypatch
):
    """Verifies that the source is NOT removed when an integrity check fails."""
    source = sample_directory_structure
    destination = tmp_path / "dest_corrupted_no_removal"

    # Mocks checksum to simulate corruption
    original_calculate_checksum = calculate_directory_checksum

    def mock_calculate_checksum(directory, **kwargs):
        result = original_calculate_checksum(directory=directory, **kwargs)
        if directory == destination:
            return "different_checksum_1234567890abcd"
        return result

    monkeypatch.setattr(
        "ataraxis_data_structures.processing.transfer_tools.calculate_directory_checksum", mock_calculate_checksum
    )

    # Attempts transfer with both verification and removal enabled
    with pytest.raises(RuntimeError):
        transfer_directory(
            source=source,
            destination=destination,
            verify_integrity=True,
            remove_source=True,
        )

    # Verifies the source still exists (was not removed due to failed verification)
    assert source.exists()
    assert (source / "file1.txt").exists()


def test_transfer_directory_integrity_check_with_progress(sample_directory_structure, tmp_path):
    """Verifies that integrity verification works with progress tracking enabled."""
    source = sample_directory_structure
    destination = tmp_path / "dest_progress_integrity"

    # Performs transfer with both progress and integrity enabled
    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
        progress=True,
    )

    # Verifies successful transfer
    assert destination.exists()
    assert (destination / "file1.txt").exists()
    assert (destination / "subdir1" / "file3.txt").exists()

    # Verifies the checksum file exists
    assert (source / "ax_checksum.txt").exists()


def test_transfer_directory_creates_checksum_when_missing(tmp_path):
    """Verifies that checksum is automatically created if missing when verify_integrity=True."""
    # Creates the source without a pre-calculated checksum
    source = tmp_path / "source_no_checksum"
    source.mkdir()
    (source / "file1.txt").write_text("content1")
    (source / "file2.txt").write_text("content2")

    destination = tmp_path / "dest_auto_checksum"

    # Verifies no checksum exists initially
    assert not (source / "ax_checksum.txt").exists()

    # Performs transfer with integrity verification
    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
    )

    # Verifies checksum was automatically created
    assert (source / "ax_checksum.txt").exists()

    # Verifies successful transfer
    assert destination.exists()
    assert (destination / "file1.txt").read_text() == "content1"


def test_transfer_directory_preserves_checksum_file(sample_directory_structure, tmp_path):
    """Verifies that the original checksum file is preserved in the source."""
    source = sample_directory_structure
    destination = tmp_path / "dest_checksum_preserved"

    # Verifies no checksum initially
    assert not (source / "ax_checksum.txt").exists()

    # Performs transfer with integrity verification
    transfer_directory(
        source=source,
        destination=destination,
        verify_integrity=True,
    )

    # Verifies the checksum file exists and persists in the source
    assert (source / "ax_checksum.txt").exists()

    # Verifies the checksum file is readable and valid
    checksum_content = (source / "ax_checksum.txt").read_text().strip()
    assert len(checksum_content) == 32  # xxHash3-128 hex string
    assert all(c in "0123456789abcdef" for c in checksum_content)


def test_transfer_directory_integrity_multithread_consistency(large_directory_structure, tmp_path):
    """Verifies that integrity checking works correctly with multithreaded transfers."""
    source = large_directory_structure
    destination = tmp_path / "dest_multi_integrity"

    # Performs multithreaded transfer with integrity verification
    transfer_directory(
        source=source,
        destination=destination,
        num_threads=4,
        verify_integrity=True,
    )

    # Verifies all files transferred correctly
    source_files = sorted([f.relative_to(source) for f in source.rglob("*.txt")])
    dest_files = sorted([f.relative_to(destination) for f in destination.rglob("*.txt")])

    assert source_files == dest_files

    # Spot checks file contents
    assert (destination / "file_0.txt").exists()
    assert (destination / "subdir_0" / "file_0.txt").exists()

    # Verifies the source checksum file was created
    assert (source / "ax_checksum.txt").exists()
