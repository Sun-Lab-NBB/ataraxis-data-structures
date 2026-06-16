"""Contains tests for the checksum_tools module provided by the processing package."""

import os
from pathlib import Path
import multiprocessing

import pytest

from ataraxis_data_structures import calculate_directory_checksum

multiprocessing.set_start_method("spawn", force=True)  # Improves reproducibility.


@pytest.fixture
def sample_directory_structure(tmp_path: Path) -> Path:
    """Creates a sample directory structure for testing."""
    root = tmp_path / "test_source"
    root.mkdir()

    (root / "file1.txt").write_text("content1")
    (root / "file2.txt").write_text("content2")

    first_subdirectory = root / "subdir1"
    first_subdirectory.mkdir()
    (first_subdirectory / "file3.txt").write_text("content3")
    (first_subdirectory / "file4.txt").write_text("content4")

    second_subdirectory = root / "subdir2"
    second_subdirectory.mkdir()
    (second_subdirectory / "file5.txt").write_text("content5")

    nested = first_subdirectory / "nested"
    nested.mkdir()
    (nested / "file6.txt").write_text("content6")

    return root


def test_calculate_directory_checksum_basic(sample_directory_structure: Path) -> None:
    """Verifies basic checksum calculation functionality."""
    checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)

    # Verifies checksum is a valid hex string.
    assert isinstance(checksum, str)
    assert len(checksum) == 32  # xxHash3-128 produces 128-bit = 32 hex chars.
    assert all(character in "0123456789abcdef" for character in checksum)


def test_calculate_directory_checksum_saves_file(sample_directory_structure: Path) -> None:
    """Verifies that the checksum file is saved when save_checksum=True."""
    checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=True)

    # Verifies checksum file exists.
    checksum_file = sample_directory_structure / "ax_checksum.txt"
    assert checksum_file.exists()

    # Verifies file content matches returned checksum.
    saved_checksum = checksum_file.read_text().strip()
    assert saved_checksum == checksum


def test_calculate_directory_checksum_consistency(sample_directory_structure: Path) -> None:
    """Verifies that calculating checksum multiple times produces identical results."""
    first_checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)
    second_checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)
    third_checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=False)

    assert first_checksum == second_checksum == third_checksum


@pytest.mark.parametrize("num_processes", [1, 2, 4, None])
def test_calculate_directory_checksum_multiprocessing(
    sample_directory_structure: Path, num_processes: int | None
) -> None:
    """Verifies that checksum calculation produces consistent results with different process counts."""
    checksum = calculate_directory_checksum(
        directory=sample_directory_structure, num_processes=num_processes, save_checksum=False
    )

    # Verifies checksum is valid.
    assert isinstance(checksum, str)
    assert len(checksum) == 32

    # Verifies consistency across different process counts by comparing with a single process.
    checksum_single = calculate_directory_checksum(
        directory=sample_directory_structure, num_processes=1, save_checksum=False
    )
    assert checksum == checksum_single


@pytest.mark.parametrize("progress", [True, False])
def test_calculate_directory_checksum_progress_mode(
    sample_directory_structure: Path,
    progress: bool,  # noqa: FBT001 - Parametrized pytest fixture value, not a positional boolean flag.
) -> None:
    """Verifies that progress mode produces identical checksums (only affects progress display)."""
    checksum = calculate_directory_checksum(
        directory=sample_directory_structure, progress=progress, save_checksum=False
    )

    # Verifies checksum is valid.
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_excludes_default_service_files(tmp_path: Path) -> None:
    """Verifies that the default excluded file (ax_checksum.txt) is excluded from checksum calculation."""
    directory_with_service_file = tmp_path / "test_exclude"
    directory_with_service_file.mkdir()

    (directory_with_service_file / "regular_file.txt").write_text("content")
    (directory_with_service_file / "ax_checksum.txt").write_text("should_be_excluded")

    checksum_with_service = calculate_directory_checksum(directory=directory_with_service_file, save_checksum=False)

    # Creates identical directory without service files.
    directory_without_service_file = tmp_path / "test_no_service"
    directory_without_service_file.mkdir()
    (directory_without_service_file / "regular_file.txt").write_text("content")

    checksum_without_service = calculate_directory_checksum(
        directory=directory_without_service_file, save_checksum=False
    )

    # Verifies checksums match (service files were excluded).
    assert checksum_with_service == checksum_without_service


def test_calculate_directory_checksum_custom_excluded_files(tmp_path: Path) -> None:
    """Verifies that a custom excluded_files set is respected."""
    directory_with_cache = tmp_path / "test_custom_exclude"
    directory_with_cache.mkdir()

    (directory_with_cache / "data.txt").write_text("content")
    (directory_with_cache / "metadata.json").write_text("{}")
    (directory_with_cache / "cache.tmp").write_text("temporary")

    # Checksum excluding cache.tmp.
    checksum_excluding = calculate_directory_checksum(
        directory=directory_with_cache, save_checksum=False, excluded_files={"cache.tmp", "ax_checksum.txt"}
    )

    # Creates identical directory without the excluded file.
    directory_without_cache = tmp_path / "test_no_cache"
    directory_without_cache.mkdir()
    (directory_without_cache / "data.txt").write_text("content")
    (directory_without_cache / "metadata.json").write_text("{}")

    checksum_without_cache = calculate_directory_checksum(
        directory=directory_without_cache, save_checksum=False, excluded_files={"ax_checksum.txt"}
    )

    assert checksum_excluding == checksum_without_cache


def test_calculate_directory_checksum_empty_excluded_files(tmp_path: Path) -> None:
    """Verifies that an empty excluded_files set includes all files."""
    directory_with_checksum_file = tmp_path / "test_empty_exclude"
    directory_with_checksum_file.mkdir()

    (directory_with_checksum_file / "data.txt").write_text("content")
    (directory_with_checksum_file / "ax_checksum.txt").write_text("included_now")

    # With empty excluded set, ax_checksum.txt is included.
    checksum_all = calculate_directory_checksum(
        directory=directory_with_checksum_file, save_checksum=False, excluded_files=set()
    )

    # Without ax_checksum.txt file.
    directory_without_checksum_file = tmp_path / "test_no_checksum_file"
    directory_without_checksum_file.mkdir()
    (directory_without_checksum_file / "data.txt").write_text("content")

    checksum_without = calculate_directory_checksum(
        directory=directory_without_checksum_file, save_checksum=False, excluded_files=set()
    )

    # They should differ since ax_checksum.txt is now included.
    assert checksum_all != checksum_without


def test_calculate_directory_checksum_empty_directory(tmp_path: Path) -> None:
    """Verifies checksum calculation for an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    checksum = calculate_directory_checksum(directory=empty_dir, save_checksum=False)

    # Verifies a valid checksum is still generated.
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_content_sensitivity(tmp_path: Path) -> None:
    """Verifies that checksum changes when file content changes."""
    test_dir = tmp_path / "content_test"
    test_dir.mkdir()

    (test_dir / "file.txt").write_text("original content")
    original_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Modifies content.
    (test_dir / "file.txt").write_text("modified content")
    modified_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksums differ.
    assert original_checksum != modified_checksum


def test_calculate_directory_checksum_structure_sensitivity(tmp_path: Path) -> None:
    """Verifies that checksum changes when the directory structure changes."""
    test_dir = tmp_path / "structure_test"
    test_dir.mkdir()

    (test_dir / "file.txt").write_text("content")
    original_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Adds a new file.
    (test_dir / "file2.txt").write_text("content")
    expanded_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksums differ.
    assert original_checksum != expanded_checksum


def test_calculate_directory_checksum_path_sensitivity(tmp_path: Path) -> None:
    """Verifies that checksum reflects file paths (not just content)."""
    # Creates two directories with the same content but different paths.
    first_directory = tmp_path / "dir1"
    first_directory.mkdir()
    (first_directory / "path_a").mkdir()
    (first_directory / "path_a" / "file.txt").write_text("same content")

    second_directory = tmp_path / "dir2"
    second_directory.mkdir()
    (second_directory / "path_b").mkdir()
    (second_directory / "path_b" / "file.txt").write_text("same content")

    first_checksum = calculate_directory_checksum(directory=first_directory, save_checksum=False)
    second_checksum = calculate_directory_checksum(directory=second_directory, save_checksum=False)

    # Verifies checksums differ due to different paths.
    assert first_checksum != second_checksum


def test_calculate_directory_checksum_large_files(tmp_path: Path) -> None:
    """Verifies checksum calculation with large files (tests chunked reading)."""
    test_dir = tmp_path / "large_files"
    test_dir.mkdir()

    # Creates a file larger than the chunk size (8 MB chunks in implementation).
    large_content = b"x" * (10 * 1024 * 1024)  # 10 MB.
    (test_dir / "large_file.bin").write_bytes(large_content)

    initial_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is generated.
    assert isinstance(initial_checksum, str)
    assert len(initial_checksum) == 32

    # Verifies consistency.
    repeated_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)
    assert initial_checksum == repeated_checksum


def test_calculate_directory_checksum_nested_structure(tmp_path: Path) -> None:
    """Verifies checksum calculation with deeply nested directory structures."""
    test_dir = tmp_path / "nested"
    current = test_dir
    for level_index in range(5):
        current /= f"level_{level_index}"
        current.mkdir(parents=True, exist_ok=True)
        (current / f"file_{level_index}.txt").write_text(f"content_{level_index}")

    checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is valid.
    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_with_existing_checksum_file(tmp_path: Path) -> None:
    """Verifies behavior when the checksum file already exists."""
    test_dir = tmp_path / "existing_checksum"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("content")

    # Pre-creates a checksum file with wrong content.
    (test_dir / "ax_checksum.txt").write_text("old_checksum_value")

    # Calculates new checksum with save enabled.
    new_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=True)

    # Verifies a file is overwritten with correct checksum.
    saved_checksum = (test_dir / "ax_checksum.txt").read_text().strip()
    assert saved_checksum == new_checksum
    assert saved_checksum != "old_checksum_value"


def test_calculate_directory_checksum_different_structures(tmp_path: Path) -> None:
    """Verifies that different directory structures produce different checksums."""
    flat_directory = tmp_path / "struct1"
    flat_directory.mkdir()
    (flat_directory / "a.txt").write_text("content_a")
    (flat_directory / "b.txt").write_text("content_b")

    # Creates second structure with same files in subdirectory.
    nested_directory = tmp_path / "struct2"
    nested_directory.mkdir()
    subdir = nested_directory / "subdir"
    subdir.mkdir()
    (subdir / "a.txt").write_text("content_a")
    (subdir / "b.txt").write_text("content_b")

    flat_checksum = calculate_directory_checksum(directory=flat_directory, save_checksum=False)
    nested_checksum = calculate_directory_checksum(directory=nested_directory, save_checksum=False)

    # Verifies different structures produce different checksums.
    assert flat_checksum != nested_checksum


def test_calculate_directory_checksum_binary_files(tmp_path: Path) -> None:
    """Verifies checksum calculation with binary files."""
    test_dir = tmp_path / "binary_test"
    test_dir.mkdir()

    (test_dir / "data.bin").write_bytes(bytes(range(256)))
    (test_dir / "zeros.bin").write_bytes(b"\x00" * 1000)
    (test_dir / "random.bin").write_bytes(os.urandom(500))

    checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)

    # Verifies checksum is valid.
    assert isinstance(checksum, str)
    assert len(checksum) == 32

    # Verifies consistency.
    repeated_checksum = calculate_directory_checksum(directory=test_dir, save_checksum=False)
    assert checksum == repeated_checksum
