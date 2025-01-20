import numpy as np
import pytest
from numpy.typing import NDArray

from ataraxis_data_structures import DataLogger, LogPackage


@pytest.fixture
def sample_data() -> tuple[int, int, NDArray[np.uint8]]:
    """Provides sample data for testing the DataLogger."""
    source_id = 1
    timestamp = 1234567890
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    return source_id, timestamp, data


@pytest.mark.xdist_group(name="group1")
def test_data_logger_initialization(tmp_path):
    """Verifies the initialization of the DataLogger class with different parameters."""
    # Tests default initialization
    logger = DataLogger(output_directory=tmp_path)
    assert logger._process_count == 1
    assert logger._thread_count == 5
    assert logger._sleep_timer == 5000
    assert logger._output_directory == tmp_path / f"{logger.name}_data_log"
    assert logger._started is False
    assert len(logger._logger_processes) == 0

    # Tests custom initialization
    logger = DataLogger(output_directory=tmp_path, process_count=2, thread_count=10, sleep_timer=1000)
    assert logger._process_count == 2
    assert logger._thread_count == 10
    assert logger._sleep_timer == 1000
    print(logger)  # Ensures __repr__ works as expected


@pytest.mark.xdist_group(name="group1")
def test_data_logger_directory_creation(tmp_path):
    """Verifies that the DataLogger creates the necessary output directory."""
    logger = DataLogger(output_directory=tmp_path)
    assert logger.output_directory.exists()
    assert logger.output_directory.is_dir()


@pytest.mark.xdist_group(name="group1")
def test_data_logger_start_stop(tmp_path):
    """Verifies the start and stop functionality of the DataLogger."""
    logger = DataLogger(output_directory=tmp_path)
    assert not logger.started

    # Test start
    logger.start()
    assert logger.started
    logger.start()  # Ensures that calling start() twice does nothing.
    assert logger._started is True
    assert all(process.is_alive() for process in logger._logger_processes)

    # Tests activating multiple concurrent loggers with different instance names
    logger_2 = DataLogger(output_directory=tmp_path, instance_name="custom_name")
    logger_2.start()

    # Test stop
    logger.stop()
    assert not logger.started
    assert all(not process.is_alive() for process in logger._logger_processes)
    logger.stop()  # Verifies that calling stop twice does nothing


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "process_count, thread_count",
    [
        (1, 5),  # Default configuration
        (2, 3),  # Multiple processes, fewer threads
        (3, 10),  # More processes and threads
    ],
)
def test_data_logger_multiprocessing(tmp_path, process_count, thread_count, sample_data):
    """Verifies that DataLogger correctly handles multiple processes and threads."""
    logger = DataLogger(output_directory=tmp_path, process_count=process_count, thread_count=thread_count)
    logger.start()

    # Submit multiple data points
    for i in range(5):
        source_id, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(source_id=np.uint16(source_id), time_stamp=np.uint64(timestamp), serialized_data=data)
        logger.input_queue.put(packed_data)

    # Allow some time for processing
    logger.stop()

    # Verify files were created
    log_dir = tmp_path / f"{logger.name}_data_log"
    files = list(log_dir.glob("*.npy"))
    assert len(files) > 0


@pytest.mark.xdist_group(name="group1")
def test_data_logger_data_integrity(tmp_path, sample_data):
    """Verifies that saved data maintains integrity through the logging process."""
    logger = DataLogger(output_directory=tmp_path)
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint16(source_id), time_stamp=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    logger.stop()

    # Verify the saved file
    saved_files = list(logger.output_directory.glob("*.npy"))
    assert len(saved_files) == 1

    # Load and verify the saved data
    saved_data = np.load(saved_files[0])

    # Extract components from saved data
    saved_source_id = int.from_bytes(saved_data[:2].tobytes(), byteorder="little")
    saved_timestamp = int.from_bytes(saved_data[2:10].tobytes(), byteorder="little")
    saved_content = saved_data[10:]

    assert saved_source_id == source_id
    assert saved_timestamp == timestamp
    np.testing.assert_array_equal(saved_content, data)


@pytest.mark.xdist_group(name="group1")
def test_data_logger_compression(tmp_path, sample_data):
    """Verifies the log compression functionality."""
    logger = DataLogger(output_directory=tmp_path)
    logger.start()

    # Submit multiple data points with different source IDs
    source_ids = [1, 1, 2, 2]
    for i, source_id in enumerate(source_ids):
        _, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(source_id=np.uint16(source_id), time_stamp=np.uint64(timestamp), serialized_data=data)
        logger.input_queue.put(packed_data)

    logger.stop()

    # Test compression
    logger.compress_logs(remove_sources=True, verbose=True)

    # Verify compressed files
    compressed_files = list(tmp_path.glob("**/*.npz"))
    assert len(compressed_files) == 2  # One for each unique source_id

    # Verify original files were removed
    original_files = list(logger.output_directory.glob("*.npy"))
    assert len(original_files) == 0


@pytest.mark.xdist_group(name="group1")
def test_data_logger_concurrent_access(tmp_path, sample_data):
    """Verifies that DataLogger handles concurrent access correctly."""
    logger = DataLogger(output_directory=tmp_path, process_count=2, thread_count=5)
    logger.start()

    from concurrent.futures import ThreadPoolExecutor

    def submit_data(i):
        source_id, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(source_id=np.uint16(source_id), time_stamp=np.uint64(timestamp), serialized_data=data)
        logger.input_queue.put(packed_data)

    # Submit data concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(submit_data, range(20))

    logger.stop()

    # Verify all files were created
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 20

    # Verifies log compression with source deletion and not memory mapping
    logger.compress_logs(remove_sources=True, memory_mapping=False)
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 0
    files = list(logger.output_directory.glob("*.npz"))
    assert len(files) == 1


@pytest.mark.xdist_group(name="group1")
def test_data_logger_empty_queue_shutdown(tmp_path):
    """Verifies that DataLogger shuts down correctly with an empty queue."""
    logger = DataLogger(output_directory=tmp_path)
    logger.start()

    # Immediate stop without any data
    logger.stop()

    # Verify no files were created
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 0


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize("sleep_timer", [0, 1000, 5000])
def test_data_logger_sleep_timer(tmp_path, sleep_timer, sample_data):
    """Verifies that DataLogger respects different sleep timer settings."""
    logger = DataLogger(output_directory=tmp_path, sleep_timer=sleep_timer)
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint16(source_id), time_stamp=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    # Allow time for processing
    logger.stop()

    # Verify data was saved regardless of sleep timer
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 1
