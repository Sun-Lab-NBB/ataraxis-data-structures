"""Contains tests for classes and functions provided by the serialized_data_logger.py module."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.typing import NDArray

from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives


@pytest.fixture
def sample_data() -> tuple[int, int, NDArray[np.uint8]]:
    """Provides sample data for testing the DataLogger."""
    source_id = 1
    timestamp = 1234567890
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    return source_id, timestamp, data


@pytest.mark.xdist_group(name="group1")
def test_data_logger_initialization(tmp_path: Path) -> None:
    """Verifies the initialization of the DataLogger class with different parameters."""
    # Tests default initialization.
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert logger._thread_count == 5
    assert logger._poll_interval == 5
    assert logger._output_directory == tmp_path / "test_logger_data_log"
    assert not logger._started
    assert logger._logger_process is None
    assert logger.name == "test_logger"

    # Tests custom initialization.
    logger = DataLogger(output_directory=tmp_path, instance_name="custom_logger", thread_count=10, poll_interval=1000)
    assert logger._thread_count == 10
    assert logger._poll_interval == 1000
    # Ensures __repr__ works as expected.
    assert repr(logger)


@pytest.mark.xdist_group(name="group1")
def test_data_logger_directory_creation(tmp_path: Path) -> None:
    """Verifies that the DataLogger creates the necessary output directory."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert logger.output_directory.exists()
    assert logger.output_directory.is_dir()


@pytest.mark.xdist_group(name="group1")
def test_data_logger_start_stop(tmp_path: Path) -> None:
    """Verifies the start and stop functionality of the DataLogger."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert not logger.alive

    # Tests start.
    logger.start()
    assert logger.alive
    # Ensures that calling start() twice does nothing.
    logger.start()
    assert logger._started
    assert logger._logger_process.is_alive()

    # Tests activating multiple concurrent loggers with different instance names.
    logger_2 = DataLogger(output_directory=tmp_path, instance_name="custom_name")
    logger_2.start()

    # Tests stop.
    logger.stop()
    assert not logger.alive
    assert not logger._logger_process.is_alive()
    # Verifies that calling stop twice does nothing.
    logger.stop()

    # Cleans up the second logger.
    logger_2.stop()


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "thread_count",
    [5, 3, 10],  # Different thread configurations
)
def test_data_logger_multithreading(
    tmp_path: Path, thread_count: int, sample_data: tuple[int, int, NDArray[np.uint8]]
) -> None:
    """Verifies that DataLogger correctly handles multiple threads."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", thread_count=thread_count)
    logger.start()

    # Submits multiple data points.
    for index in range(5):
        source_id, timestamp, data = sample_data
        timestamp += index
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    # Allows time for processing.
    logger.stop()

    # Verifies files were created.
    log_directory = tmp_path / "test_logger_data_log"
    files = list(log_directory.glob("*.npy"))
    assert files


@pytest.mark.xdist_group(name="group1")
def test_data_logger_data_integrity(tmp_path: Path, sample_data: tuple[int, int, NDArray[np.uint8]]) -> None:
    """Verifies that saved data maintains integrity through the logging process."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    logger.stop()

    # Verifies the saved file.
    saved_files = list(logger.output_directory.glob("*.npy"))
    assert len(saved_files) == 1

    # Loads and verifies the saved data.
    saved_data = np.load(saved_files[0])

    # Extracts components from saved data.
    saved_source_id = int.from_bytes(saved_data[:1].tobytes(), byteorder="little")
    saved_timestamp = int.from_bytes(saved_data[1:9].tobytes(), byteorder="little")
    saved_content = saved_data[9:]

    assert saved_source_id == source_id
    assert saved_timestamp == timestamp
    np.testing.assert_array_equal(saved_content, data)


@pytest.mark.xdist_group(name="group1")
def test_data_logger_assembly(tmp_path: Path, sample_data: tuple[int, int, NDArray[np.uint8]]) -> None:
    """Verifies the log archive assembly functionality using the standalone function."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Submits multiple data points with different source IDs.
    source_ids = [1, 1, 2, 2]
    for index, source_id in enumerate(source_ids):
        _, timestamp, data = sample_data
        timestamp += index
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    logger.stop()

    # Tests log assembly using a standalone function.
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=True)

    # Verifies log archives.
    compressed_files = list(logger.output_directory.glob("*.npz"))
    assert len(compressed_files) == 2  # One for each unique source_id

    # Verifies original files were removed.
    original_files = list(logger.output_directory.glob("*.npy"))
    assert not original_files


@pytest.mark.xdist_group(name="group1")
def test_data_logger_concurrent_access(tmp_path: Path, sample_data: tuple[int, int, NDArray[np.uint8]]) -> None:
    """Verifies that DataLogger handles concurrent access correctly."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", thread_count=5)
    logger.start()

    def submit_data(index: int) -> None:
        source_id, timestamp, data = sample_data
        timestamp += index
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    # Submits data concurrently.
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(submit_data, range(20))

    logger.stop()

    # Verifies all files were created.
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 20

    # Verifies archive creation with source deletion and not memory mapping.
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, memory_mapping=False)
    files = list(logger.output_directory.glob("*.npy"))
    assert not files
    files = list(logger.output_directory.glob("*.npz"))
    assert len(files) == 1


@pytest.mark.xdist_group(name="group1")
def test_data_logger_empty_queue_shutdown(tmp_path: Path) -> None:
    """Verifies that DataLogger shuts down correctly with an empty queue."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Stops without sending any data.
    logger.stop()

    # Verifies no files were created.
    files = list(logger.output_directory.glob("*.npy"))
    assert not files


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize("poll_interval", [0, 1000, 5000])
def test_data_logger_poll_interval(
    tmp_path: Path, poll_interval: int, sample_data: tuple[int, int, NDArray[np.uint8]]
) -> None:
    """Verifies that DataLogger respects different poll interval settings."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", poll_interval=poll_interval)
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    # Allows time for processing.
    logger.stop()

    # Verifies data was saved regardless of poll interval.
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 1


@pytest.mark.xdist_group(name="group1")
def test_data_logger_start_stop_cycling(tmp_path: Path) -> None:
    """Verifies that cycling start and stop method of DataLogger does not produce errors."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()
    logger.start()
    logger.start()
    logger.stop()


@pytest.mark.xdist_group(name="group1")
def test_assemble_log_archives_with_integrity_check(
    tmp_path: Path, sample_data: tuple[int, int, NDArray[np.uint8]]
) -> None:
    """Verifies the integrity checking feature of assemble_log_archives."""
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Submits test data.
    for index in range(3):
        source_id, timestamp, data = sample_data
        timestamp += index
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    logger.stop()

    # Tests archive assembly with integrity verification.
    assemble_log_archives(
        log_directory=logger.output_directory, remove_sources=False, verify_integrity=True, verbose=False
    )

    # Verifies both original and archive files exist.
    original_files = list(logger.output_directory.glob("*.npy"))
    compressed_files = list(logger.output_directory.glob("*.npz"))
    assert len(original_files) == 3
    assert len(compressed_files) == 1


def test_log_package_data_golden_bytes() -> None:
    """Verifies that LogPackage.data serializes the header and payload to the exact byte layout consumers depend on."""
    # The on-disk layout is a fixed contract shared with LogArchiveReader and downstream parsers:
    # [source_id (1 byte)][acquisition_time (8 bytes, little-endian uint64)][payload (N bytes)].
    source_id = np.uint8(7)
    acquisition_time = np.uint64(1234567890)
    payload = np.array([10, 20, 30, 255], dtype=np.uint8)

    log_name, data = LogPackage(source_id=source_id, acquisition_time=acquisition_time, serialized_data=payload).data

    # Hardcoded golden bytes pin the format independently of any serialization helper.
    expected = bytes([7]) + (1234567890).to_bytes(length=8, byteorder="little") + bytes([10, 20, 30, 255])
    assert data.dtype == np.uint8
    assert data.tobytes() == expected

    # Verifies the zero-padded filename format.
    assert log_name == "007_00000000001234567890.npy"

    # Verifies the layout round-trips exactly as LogArchiveReader reads it.
    assert int(data[0]) == 7
    assert int(data[1:9].view(np.uint64)[0]) == 1234567890
    np.testing.assert_array_equal(data[9:], payload)


def test_log_package_data_large_timestamp() -> None:
    """Verifies that LogPackage.data serializes uint64 timestamps at or above 2**63 without overflow or truncation."""
    source_id = np.uint8(255)
    acquisition_time = np.uint64(2**63 + 5)  # Exceeds the signed int64 range.
    payload = np.array([1], dtype=np.uint8)

    _, data = LogPackage(source_id=source_id, acquisition_time=acquisition_time, serialized_data=payload).data

    expected = bytes([255]) + (2**63 + 5).to_bytes(length=8, byteorder="little") + bytes([1])
    assert data.tobytes() == expected
    assert int(data[1:9].view(np.uint64)[0]) == 2**63 + 5
