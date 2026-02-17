"""Contains tests for the checksum_tools module provided by the processing package."""

import os
from pathlib import Path
import multiprocessing

import pytest

from ataraxis_data_structures import calculate_directory_checksum

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


def test_calculate_directory_checksum_basic(sample_directory_structure):
    """Verifies basic checksum calculation functionality."""
    checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)

    # Verifies checksum is a valid hex string
    assert isinstance(checksum, str)
    assert len(checksum) == 32  # xxHash3-128 produces 128-bit = 32 hex chars
    assert all(c in "0123456789abcdef" for c in checksum)


def test_calculate_directory_checksum_saves_file(sample_directory_structure):
    """Verifies that the checksum file is saved when save_checksum=True."""
    checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=True)

    # Verifies checksum file exists
    checksum_file = sample_directory_structure / "ax_checksum.txt"
    assert checksum_file.exists()

    # Verifies file content matches returned checksum
    saved_checksum = checksum_file.read_text().strip()
    assert saved_checksum == checksum


def test_calculate_directory_checksum_consistency(sample_directory_structure):
    """Verifies that calculating checksum multiple times produces identical results."""
    checksum1 = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)
    checksum2 = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)
    checksum3 = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)

    assert checksum1 == checksum2 == checksum3


@pytest.mark.parametrize("num_processes", [1, 2, 4, None])
def test_calculate_directory_checksum_multiprocessing(sample_directory_structure, num_processes):
    """Verifies that checksum calculation produces consistent results with different process counts."""
    checksum = calculate_directory_checksum(
        directory=sample_directory_structure, num_processes=num_processes, save_checksum=False
    )

    # Verifies checksum is valid
    assert isinstance(checksum, str)
    assert len(checksum) == 32

    # Verifies consistency across different process counts by comparing with a single process
    checksum_single = calculate_directory_checksum(
        directory=sample_directory_structure, num_processes=1, save_checksum=False
    )
    assert checksum == checksum_single


@pytest.mark.parametrize("progress", [True, False])
def test_calculate_directory_checksum_progress_mode(sample_directory_structure, progress):
    """Verifies that progress mode produces identical checksums (only affects progress display)."""
    checksum = calculate_directory_checksum(
        directory=sample_directory_structure, progress=progress, save_checksum=False
    )

    # Verifies checksum is valid
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_excludes_default_service_files(tmp_path):
    """Verifies that the default excluded file (ax_checksum.txt) is excluded from checksum calculation."""
    # Creates a directory with regular and service files
    test_dir = tmp_path / "test_exclude"
    test_dir.mkdir()

    (test_dir / "regular_file.txt").write_text("content")
    (test_dir / "ax_checksum.txt").write_text("should_be_excluded")

    checksum_with_service = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Creates identical directory without service files
    test_dir2 = tmp_path / "test_no_service"
    test_dir2.mkdir()
    (test_dir2 / "regular_file.txt").write_text("content")

    checksum_without_service = calculate_directory_checksum(directory=test_dir2, save_checksum=False)

    # Verifies checksums match (service files were excluded)
    assert checksum_with_service == checksum_without_service


def test_calculate_directory_checksum_custom_excluded_files(tmp_path):
    """Verifies that a custom excluded_files set is respected."""
    test_dir = tmp_path / "test_custom_exclude"
    test_dir.mkdir()

    (test_dir / "data.txt").write_text("content")
    (test_dir / "metadata.json").write_text("{}")
    (test_dir / "cache.tmp").write_text("temporary")

    # Checksum excluding cache.tmp
    checksum_excluding = calculate_directory_checksum(
        directory=test_dir, save_checksum=False, excluded_files={"cache.tmp", "ax_checksum.txt"}
    )

    # Creates identical directory without the excluded file
    test_dir2 = tmp_path / "test_no_cache"
    test_dir2.mkdir()
    (test_dir2 / "data.txt").write_text("content")
    (test_dir2 / "metadata.json").write_text("{}")

    checksum_without_cache = calculate_directory_checksum(
        directory=test_dir2, save_checksum=False, excluded_files={"ax_checksum.txt"}
    )

    assert checksum_excluding == checksum_without_cache


def test_calculate_directory_checksum_empty_excluded_files(tmp_path):
    """Verifies that an empty excluded_files set includes all files."""
    test_dir = tmp_path / "test_empty_exclude"
    test_dir.mkdir()

    (test_dir / "data.txt").write_text("content")
    (test_dir / "ax_checksum.txt").write_text("included_now")

    # With empty excluded set, ax_checksum.txt is included
    checksum_all = calculate_directory_checksum(directory=test_dir, save_checksum=False, excluded_files=set())

    # Without ax_checksum.txt file
    test_dir2 = tmp_path / "test_no_checksum_file"
    test_dir2.mkdir()
    (test_dir2 / "data.txt").write_text("content")

    checksum_without = calculate_directory_checksum(directory=test_dir2, save_checksum=False, excluded_files=set())

    # They should differ since ax_checksum.txt is now included
    assert checksum_all != checksum_without


def test_calculate_directory_checksum_empty_directory(tmp_path):
    """Verifies checksum calculation for an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    checksum = calculate_directory_checksum(directory=empty_dir, save_checksum=False)

    # Verifies a valid checksum is still generated
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_content_sensitivity(tmp_path):
    """Verifies that checksum changes when file content changes."""
    test_dir = tmp_path / "content_test"
    test_dir.mkdir()

    # Creates the initial file
    (test_dir / "file.txt").write_text("original content")
    checksum1 = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Modifies content
    (test_dir / "file.txt").write_text("modified content")
    checksum2 = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksums differ
    assert checksum1 != checksum2


def test_calculate_directory_checksum_structure_sensitivity(tmp_path):
    """Verifies that checksum changes when the directory structure changes."""
    test_dir = tmp_path / "structure_test"
    test_dir.mkdir()

    # Creates initial structure
    (test_dir / "file.txt").write_text("content")
    checksum1 = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Adds a new file
    (test_dir / "file2.txt").write_text("content")
    checksum2 = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksums differ
    assert checksum1 != checksum2


def test_calculate_directory_checksum_path_sensitivity(tmp_path):
    """Verifies that checksum reflects file paths (not just content)."""
    # Creates two directories with the same content but different paths
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "path_a").mkdir()
    (dir1 / "path_a" / "file.txt").write_text("same content")

    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    (dir2 / "path_b").mkdir()
    (dir2 / "path_b" / "file.txt").write_text("same content")

    checksum1 = calculate_directory_checksum(directory=dir1, save_checksum=False)
    checksum2 = calculate_directory_checksum(directory=dir2, save_checksum=False)

    # Verifies checksums differ due to different paths
    assert checksum1 != checksum2


def test_calculate_directory_checksum_large_files(tmp_path):
    """Verifies checksum calculation with large files (tests chunked reading)."""
    test_dir = tmp_path / "large_files"
    test_dir.mkdir()

    # Creates a file larger than the chunk size (8 MB chunks in implementation)
    large_content = b"x" * (10 * 1024 * 1024)  # 10 MB
    (test_dir / "large_file.bin").write_bytes(large_content)

    checksum1 = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is generated
    assert isinstance(checksum1, str)
    assert len(checksum1) == 32

    # Verifies consistency
    checksum2 = calculate_directory_checksum(directory=test_dir, save_checksum=False)
    assert checksum1 == checksum2


def test_calculate_directory_checksum_nested_structure(tmp_path):
    """Verifies checksum calculation with deeply nested directory structures."""
    # Creates a deeply nested structure
    test_dir = tmp_path / "nested"
    current = test_dir
    for i in range(5):
        current /= f"level_{i}"
        current.mkdir(parents=True, exist_ok=True)
        (current / f"file_{i}.txt").write_text(f"content_{i}")

    checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is valid
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_with_existing_checksum_file(tmp_path):
    """Verifies behavior when the checksum file already exists."""
    test_dir = tmp_path / "existing_checksum"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")

    # Pre-creates a checksum file with wrong content
    (test_dir / "ax_checksum.txt").write_text("old_checksum_value")

    # Calculates new checksum with save enabled
    new_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=True)

    # Verifies a file is overwritten with correct checksum
    saved_checksum = (test_dir / "ax_checksum.txt").read_text().strip()
    assert saved_checksum == new_checksum
    assert saved_checksum != "old_checksum_value"


def test_calculate_directory_checksum_different_structures(tmp_path):
    """Verifies that different directory structures produce different checksums."""
    # Creates first structure
    dir1 = tmp_path / "struct1"
    dir1.mkdir()
    (dir1 / "a.txt").write_text("content_a")
    (dir1 / "b.txt").write_text("content_b")

    # Creates second structure with same files in subdirectory
    dir2 = tmp_path / "struct2"
    dir2.mkdir()
    subdir = dir2 / "subdir"
    subdir.mkdir()
    (subdir / "a.txt").write_text("content_a")
    (subdir / "b.txt").write_text("content_b")

    checksum1 = calculate_directory_checksum(directory=dir1, save_checksum=False)
    checksum2 = calculate_directory_checksum(directory=dir2, save_checksum=False)

    # Verifies different structures produce different checksums
    assert checksum1 != checksum2


def test_calculate_directory_checksum_binary_files(tmp_path):
    """Verifies checksum calculation with binary files."""
    test_dir = tmp_path / "binary_test"
    test_dir.mkdir()

    # Creates various binary files
    (test_dir / "data.bin").write_bytes(bytes(range(256)))
    (test_dir / "zeros.bin").write_bytes(b"\x00" * 1000)
    (test_dir / "random.bin").write_bytes(os.urandom(500))

    checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is valid
    assert isinstance(checksum, str)
    assert len(checksum) == 32

    # Verifies consistency
    checksum2 = calculate_directory_checksum(directory=test_dir, save_checksum=False)
    assert checksum == checksum2
