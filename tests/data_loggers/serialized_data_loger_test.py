import numpy as np
import pytest
from numpy.typing import NDArray

from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives


@pytest.fixture
def sample_data() -> tuple[int, int, NDArray[np.uint8]]:
    """Provides sample data for testing the DataLogger.

    Returns:
        A tuple containing source_id, timestamp, and data array for testing.
    """
    source_id = 1
    timestamp = 1234567890
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    return source_id, timestamp, data


@pytest.mark.xdist_group(name="group1")
def test_data_logger_initialization(tmp_path):
    """Verifies the initialization of the DataLogger class with different parameters.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    # Tests default initialization
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert logger._thread_count == 5
    assert logger._poll_interval == 5
    assert logger._output_directory == tmp_path / "test_logger_data_log"
    assert logger._started is False
    assert logger._logger_process is None

    # Tests custom initialization
    logger = DataLogger(output_directory=tmp_path, instance_name="custom_logger", thread_count=10, poll_interval=1000)
    assert logger._thread_count == 10
    assert logger._poll_interval == 1000
    print(logger)  # Ensures __repr__ works as expected


@pytest.mark.xdist_group(name="group1")
def test_data_logger_directory_creation(tmp_path):
    """Verifies that the DataLogger creates the necessary output directory.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert logger.output_directory.exists()
    assert logger.output_directory.is_dir()


@pytest.mark.xdist_group(name="group1")
def test_data_logger_start_stop(tmp_path):
    """Verifies the start and stop functionality of the DataLogger.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    assert not logger.started

    # Tests start
    logger.start()
    assert logger.started
    logger.start()  # Ensures that calling start() twice does nothing
    assert logger._started is True
    assert logger._logger_process.is_alive()

    # Tests activating multiple concurrent loggers with different instance names
    logger_2 = DataLogger(output_directory=tmp_path, instance_name="custom_name")
    logger_2.start()

    # Tests stop
    logger.stop()
    assert not logger.started
    assert not logger._logger_process.is_alive()
    logger.stop()  # Verifies that calling stop twice does nothing

    # Cleans up the second logger
    logger_2.stop()


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize(
    "thread_count",
    [5, 3, 10],  # Different thread configurations
)
def test_data_logger_multithreading(tmp_path, thread_count, sample_data):
    """Verifies that DataLogger correctly handles multiple threads.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        thread_count: Number of threads to test.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", thread_count=thread_count)
    logger.start()

    # Submits multiple data points
    for i in range(5):
        source_id, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    # Allows time for processing
    logger.stop()

    # Verifies files were created
    log_dir = tmp_path / "test_logger_data_log"
    files = list(log_dir.glob("*.npy"))
    assert len(files) > 0


@pytest.mark.xdist_group(name="group1")
def test_data_logger_data_integrity(tmp_path, sample_data):
    """Verifies that saved data maintains integrity through the logging process.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    logger.stop()

    # Verifies the saved file
    saved_files = list(logger.output_directory.glob("*.npy"))
    assert len(saved_files) == 1

    # Loads and verifies the saved data
    saved_data = np.load(saved_files[0])

    # Extracts components from saved data
    saved_source_id = int.from_bytes(saved_data[:1].tobytes(), byteorder="little")
    saved_timestamp = int.from_bytes(saved_data[1:9].tobytes(), byteorder="little")
    saved_content = saved_data[9:]

    assert saved_source_id == source_id
    assert saved_timestamp == timestamp
    np.testing.assert_array_equal(saved_content, data)


@pytest.mark.xdist_group(name="group1")
def test_data_logger_assembly(tmp_path, sample_data):
    """Verifies the log archive assembly functionality using the standalone function.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Submits multiple data points with different source IDs
    source_ids = [1, 1, 2, 2]
    for i, source_id in enumerate(source_ids):
        _, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    logger.stop()

    # Tests log assembly using a standalone function
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=True)

    # Verifies log archives
    compressed_files = list(logger.output_directory.glob("*.npz"))
    assert len(compressed_files) == 2  # One for each unique source_id

    # Verifies original files were removed
    original_files = list(logger.output_directory.glob("*.npy"))
    assert len(original_files) == 0


@pytest.mark.xdist_group(name="group1")
def test_data_logger_concurrent_access(tmp_path, sample_data):
    """Verifies that DataLogger handles concurrent access correctly.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", thread_count=5)
    logger.start()

    from concurrent.futures import ThreadPoolExecutor

    def submit_data(i):
        source_id, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    # Submits data concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(submit_data, range(20))

    logger.stop()

    # Verifies all files were created
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 20

    # Verifies archive creation with source deletion and not memory mapping
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, memory_mapping=False)
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 0
    files = list(logger.output_directory.glob("*.npz"))
    assert len(files) == 1


@pytest.mark.xdist_group(name="group1")
def test_data_logger_empty_queue_shutdown(tmp_path):
    """Verifies that DataLogger shuts down correctly with an empty queue.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Stops without sending any data
    logger.stop()

    # Verifies no files were created
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 0


@pytest.mark.xdist_group(name="group1")
@pytest.mark.parametrize("poll_interval", [0, 1000, 5000])
def test_data_logger_poll_interval(tmp_path, poll_interval, sample_data):
    """Verifies that DataLogger respects different poll interval settings.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        poll_interval: Poll interval in milliseconds to test.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger", poll_interval=poll_interval)
    logger.start()

    source_id, timestamp, data = sample_data
    packed_data = LogPackage(source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data)
    logger.input_queue.put(packed_data)

    # Allows time for processing
    logger.stop()

    # Verifies data was saved regardless of poll interval
    files = list(logger.output_directory.glob("*.npy"))
    assert len(files) == 1


@pytest.mark.xdist_group(name="group1")
def test_data_logger_start_stop_cycling(tmp_path):
    """Verifies that cycling start and stop method of DataLogger does not produce errors.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()
    logger.start()
    logger.start()
    logger.stop()


@pytest.mark.xdist_group(name="group1")
def test_assemble_log_archives_with_integrity_check(tmp_path, sample_data):
    """Verifies the integrity checking feature of assemble_log_archives.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.
        sample_data: Sample data fixture for testing.
    """
    logger = DataLogger(output_directory=tmp_path, instance_name="test_logger")
    logger.start()

    # Submits test data
    for i in range(3):
        source_id, timestamp, data = sample_data
        timestamp += i
        packed_data = LogPackage(
            source_id=np.uint8(source_id), acquisition_time=np.uint64(timestamp), serialized_data=data
        )
        logger.input_queue.put(packed_data)

    logger.stop()

    # Tests archive assembly with integrity verification
    assemble_log_archives(
        log_directory=logger.output_directory, remove_sources=False, verify_integrity=True, verbose=False
    )

    # Verifies both original and archive files exist
    original_files = list(logger.output_directory.glob("*.npy"))
    compressed_files = list(logger.output_directory.glob("*.npz"))
    assert len(original_files) == 3
    assert len(compressed_files) == 1
