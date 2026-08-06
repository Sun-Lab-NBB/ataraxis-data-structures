"""Contains tests for the checksum_tools module provided by the processing package."""

import os
from pathlib import Path

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import calculate_directory_checksum


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

    assert isinstance(checksum, str)
    assert len(checksum) == 32  # xxHash3-128 produces 128-bit = 32 hex characters.
    assert all(character in "0123456789abcdef" for character in checksum)


def test_calculate_directory_checksum_saves_file(sample_directory_structure: Path) -> None:
    """Verifies that the checksum file is saved when save_checksum=True."""
    checksum = calculate_directory_checksum(directory=sample_directory_structure, save_checksum=True)

    checksum_file = sample_directory_structure / "ax_checksum.txt"
    assert checksum_file.exists()

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
    """Verifies that a checksum calculated with progress tracking enabled is a valid digest."""
    checksum = calculate_directory_checksum(
        directory=sample_directory_structure, progress=progress, save_checksum=False
    )

    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_excludes_default_service_files(tmp_path: Path) -> None:
    """Verifies that the default excluded file (ax_checksum.txt) is excluded from checksum calculation."""
    directory_with_service_file = tmp_path / "test_exclude"
    directory_with_service_file.mkdir()

    (directory_with_service_file / "regular_file.txt").write_text("content")
    (directory_with_service_file / "ax_checksum.txt").write_text("should_be_excluded")

    checksum_with_service = calculate_directory_checksum(directory=directory_with_service_file, save_checksum=False)

    directory_without_service_file = tmp_path / "test_no_service"
    directory_without_service_file.mkdir()
    (directory_without_service_file / "regular_file.txt").write_text("content")

    checksum_without_service = calculate_directory_checksum(
        directory=directory_without_service_file, save_checksum=False
    )

    assert checksum_with_service == checksum_without_service


def test_calculate_directory_checksum_custom_excluded_files(tmp_path: Path) -> None:
    """Verifies that a custom excluded_files set is respected."""
    directory_with_cache = tmp_path / "test_custom_exclude"
    directory_with_cache.mkdir()

    (directory_with_cache / "data.txt").write_text("content")
    (directory_with_cache / "metadata.json").write_text("{}")
    (directory_with_cache / "cache.tmp").write_text("temporary")

    checksum_excluding = calculate_directory_checksum(
        directory=directory_with_cache, save_checksum=False, excluded_files={"cache.tmp", "ax_checksum.txt"}
    )

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

    checksum_all = calculate_directory_checksum(
        directory=directory_with_checksum_file, save_checksum=False, excluded_files=set()
    )

    directory_without_checksum_file = tmp_path / "test_no_checksum_file"
    directory_without_checksum_file.mkdir()
    (directory_without_checksum_file / "data.txt").write_text("content")

    checksum_without = calculate_directory_checksum(
        directory=directory_without_checksum_file, save_checksum=False, excluded_files=set()
    )

    # Differs because ax_checksum.txt now contributes to the checksum.
    assert checksum_all != checksum_without


def test_calculate_directory_checksum_rejects_an_empty_directory(tmp_path: Path) -> None:
    """Verifies that checksumming a directory holding no file is rejected."""
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()

    message = (
        f"Unable to calculate the checksum for the {empty_directory} directory. The directory must hold at least one "
        f"file the checksum can cover, but it holds none."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        calculate_directory_checksum(directory=empty_directory, save_checksum=False)


def test_calculate_directory_checksum_rejects_a_directory_holding_only_excluded_files(tmp_path: Path) -> None:
    """Verifies that a directory whose every file is excluded is rejected, since nothing remains to cover."""
    directory = tmp_path / "only_excluded"
    directory.mkdir()
    (directory / "ax_checksum.txt").write_text("stale")

    message = (
        f"Unable to calculate the checksum for the {directory} directory. The directory must hold at least one "
        f"file the checksum can cover, but it holds none."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        calculate_directory_checksum(directory=directory, save_checksum=False)


def test_calculate_directory_checksum_matches_a_recorded_digest(tmp_path: Path) -> None:
    """Verifies that a fixed tree produces the exact digest earlier releases produced for it."""
    directory = tmp_path / "golden"
    directory.mkdir()
    (directory / "data.txt").write_text("payload")
    (directory / ".gitignore").write_text("ignored")
    (directory / "archive.tar.gz").write_text("compressed")

    # Pins the digest itself rather than a relationship between two digests, so a change to which files discovery
    # reports, or to how each one is folded in, fails here instead of shipping. The tree is flat, since a nested path
    # renders with the host separator and would make the recorded value platform-dependent.
    assert calculate_directory_checksum(directory=directory, save_checksum=False) == "697ab93b2e5603ddc5ba5556a7f3954c"


def test_calculate_directory_checksum_omits_an_empty_subdirectory(tmp_path: Path) -> None:
    """Verifies that an empty subdirectory inside a populated tree is omitted rather than rejected."""
    directory = tmp_path / "populated"
    directory.mkdir()
    (directory / "file.txt").write_text("content")

    without_subdirectory = calculate_directory_checksum(directory=directory, save_checksum=False)
    (directory / "empty_subdirectory").mkdir()
    with_subdirectory = calculate_directory_checksum(directory=directory, save_checksum=False)

    assert with_subdirectory == without_subdirectory


def test_calculate_directory_checksum_content_sensitivity(tmp_path: Path) -> None:
    """Verifies that checksum changes when file content changes."""
    test_directory = tmp_path / "content_test"
    test_directory.mkdir()

    (test_directory / "file.txt").write_text("original content")
    original_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    (test_directory / "file.txt").write_text("modified content")
    modified_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    assert original_checksum != modified_checksum


def test_calculate_directory_checksum_structure_sensitivity(tmp_path: Path) -> None:
    """Verifies that checksum changes when the directory structure changes."""
    test_directory = tmp_path / "structure_test"
    test_directory.mkdir()

    (test_directory / "file.txt").write_text("content")
    original_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    (test_directory / "file2.txt").write_text("content")
    expanded_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    assert original_checksum != expanded_checksum


def test_calculate_directory_checksum_path_sensitivity(tmp_path: Path) -> None:
    """Verifies that the checksum reflects file paths in addition to file content."""
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

    assert first_checksum != second_checksum


def test_calculate_directory_checksum_large_files(tmp_path: Path) -> None:
    """Verifies checksum calculation with large files (tests chunked reading)."""
    test_directory = tmp_path / "large_files"
    test_directory.mkdir()

    # Creates a file larger than the chunk size (8 MB chunks in implementation).
    large_content = b"x" * (10 * 1024 * 1024)  # 10 MB.
    (test_directory / "large_file.bin").write_bytes(large_content)

    initial_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    assert isinstance(initial_checksum, str)
    assert len(initial_checksum) == 32

    repeated_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)
    assert initial_checksum == repeated_checksum


def test_calculate_directory_checksum_nested_structure(tmp_path: Path) -> None:
    """Verifies checksum calculation with deeply nested directory structures."""
    test_directory = tmp_path / "nested"
    current = test_directory
    for level_index in range(5):
        current /= f"level_{level_index}"
        current.mkdir(parents=True, exist_ok=True)
        (current / f"file_{level_index}.txt").write_text(f"content_{level_index}")

    checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    assert isinstance(checksum, str)
    assert len(checksum) == 32


def test_calculate_directory_checksum_with_existing_checksum_file(tmp_path: Path) -> None:
    """Verifies behavior when the checksum file already exists."""
    test_directory = tmp_path / "existing_checksum"
    test_directory.mkdir()
    (test_directory / "file.txt").write_text("content")

    (test_directory / "ax_checksum.txt").write_text("old_checksum_value")

    new_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=True)

    saved_checksum = (test_directory / "ax_checksum.txt").read_text().strip()
    assert saved_checksum == new_checksum
    assert saved_checksum != "old_checksum_value"


def test_calculate_directory_checksum_different_structures(tmp_path: Path) -> None:
    """Verifies that different directory structures produce different checksums."""
    flat_directory = tmp_path / "struct1"
    flat_directory.mkdir()
    (flat_directory / "a.txt").write_text("content_a")
    (flat_directory / "b.txt").write_text("content_b")

    nested_directory = tmp_path / "struct2"
    nested_directory.mkdir()
    subdirectory = nested_directory / "subdir"
    subdirectory.mkdir()
    (subdirectory / "a.txt").write_text("content_a")
    (subdirectory / "b.txt").write_text("content_b")

    flat_checksum = calculate_directory_checksum(directory=flat_directory, save_checksum=False)
    nested_checksum = calculate_directory_checksum(directory=nested_directory, save_checksum=False)

    assert flat_checksum != nested_checksum


def test_calculate_directory_checksum_binary_files(tmp_path: Path) -> None:
    """Verifies checksum calculation with binary files."""
    test_directory = tmp_path / "binary_test"
    test_directory.mkdir()

    (test_directory / "data.bin").write_bytes(bytes(range(256)))
    (test_directory / "zeros.bin").write_bytes(b"\x00" * 1000)
    (test_directory / "random.bin").write_bytes(os.urandom(500))

    checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)

    assert isinstance(checksum, str)
    assert len(checksum) == 32

    repeated_checksum = calculate_directory_checksum(directory=test_directory, save_checksum=False)
    assert checksum == repeated_checksum
