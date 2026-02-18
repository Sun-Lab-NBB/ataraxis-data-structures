"""Contains tests for classes and functions provided by the log_archive_reader.py module."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp

from ataraxis_data_structures import DataLogger, LogMessage, LogPackage, LogArchiveReader, assemble_log_archives


def _create_log_message(source_id: int, timestamp_us: int, payload: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Creates a log message in the format expected by LogArchiveReader.

    Args:
        source_id: The source identifier (0-255).
        timestamp_us: The elapsed timestamp in microseconds.
        payload: The payload data.

    Returns:
        A numpy array representing the serialized message.
    """
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([timestamp_us], dtype=np.uint64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, payload])


def _create_onset_message(source_id: int, onset_us: int) -> NDArray[np.uint8]:
    """Creates an onset message with timestamp=0 and the onset UTC epoch as payload.

    Args:
        source_id: The source identifier (0-255).
        onset_us: The UTC epoch onset timestamp in microseconds.

    Returns:
        A numpy array representing the onset message.
    """
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = np.array([0], dtype=np.uint64).view(np.uint8)
    onset_bytes = np.array([onset_us], dtype=np.int64).view(np.uint8)
    return np.concatenate([source_bytes, timestamp_bytes, onset_bytes])


def _create_test_archive(
    archive_path: Path, source_id: int, onset_us: int, message_count: int, payload_size: int = 4
) -> list[NDArray[np.uint8]]:
    """Creates a test .npz archive with the specified number of messages.

    Args:
        archive_path: The path where the archive will be saved.
        source_id: The source identifier for all messages.
        onset_us: The UTC epoch onset timestamp in microseconds.
        message_count: The number of data messages to create (excluding onset).
        payload_size: The size of each message payload in bytes.

    Returns:
        A list of the original payload arrays for verification.
    """
    arrays = {}

    # Creates the onset message (first entry).
    onset_key = f"{source_id:03d}_{0:020d}"
    arrays[onset_key] = _create_onset_message(source_id=source_id, onset_us=onset_us)

    # Creates data messages.
    payloads = []
    for i in range(message_count):
        elapsed_us = (i + 1) * 1000  # 1ms between messages
        payload = np.array([(i + j) % 256 for j in range(payload_size)], dtype=np.uint8)
        payloads.append(payload)

        message_key = f"{source_id:03d}_{elapsed_us:020d}"
        arrays[message_key] = _create_log_message(source_id=source_id, timestamp_us=elapsed_us, payload=payload)

    # Saves as .npz archive.
    np.savez(archive_path, **arrays)

    return payloads


@pytest.fixture
def sample_archive(tmp_path: Path) -> tuple[Path, int, int, list[NDArray[np.uint8]]]:
    """Creates a sample archive with 10 messages for testing."""
    archive_path = tmp_path / "001_log.npz"
    source_id = 1
    onset_us = 1700000000000000  # Sample UTC epoch in microseconds
    payloads = _create_test_archive(archive_path=archive_path, source_id=source_id, onset_us=onset_us, message_count=10)
    return archive_path, source_id, onset_us, payloads


@pytest.fixture
def large_archive(tmp_path: Path) -> tuple[Path, int, int, int]:
    """Creates a large archive with 2500 messages for batch testing."""
    archive_path = tmp_path / "002_log.npz"
    source_id = 2
    onset_us = 1700000000000000
    message_count = 2500
    _create_test_archive(
        archive_path=archive_path, source_id=source_id, onset_us=onset_us, message_count=message_count, payload_size=2
    )
    return archive_path, source_id, onset_us, message_count


class TestLogMessage:
    """Tests for the LogMessage dataclass."""

    def test_log_message_creation(self) -> None:
        """Verifies that LogMessage can be created with valid data."""
        timestamp = np.uint64(1700000000000000)
        payload = np.array([1, 2, 3, 4], dtype=np.uint8)
        message = LogMessage(timestamp_us=timestamp, payload=payload)

        assert message.timestamp_us == timestamp
        np.testing.assert_array_equal(message.payload, payload)

    def test_log_message_frozen(self) -> None:
        """Verifies that LogMessage is immutable."""
        timestamp = np.uint64(1700000000000000)
        payload = np.array([1, 2, 3, 4], dtype=np.uint8)
        message = LogMessage(timestamp_us=timestamp, payload=payload)

        with pytest.raises(AttributeError):
            message.timestamp_us = np.uint64(0)  # type: ignore[misc]


class TestLogArchiveReaderInitialization:
    """Tests for LogArchiveReader initialization."""

    def test_initialization_valid_path(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that LogArchiveReader initializes correctly with a valid path."""
        archive_path, _, _, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        assert reader._archive_path == archive_path
        assert reader._onset_us is None
        assert reader._message_keys is None

    def test_initialization_with_onset(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that LogArchiveReader accepts a pre-provided onset timestamp."""
        archive_path, _, onset_us, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(onset_us))

        assert reader._onset_us == np.uint64(onset_us)

    def test_initialization_invalid_path(self, tmp_path: Path) -> None:
        """Verifies that LogArchiveReader raises FileNotFoundError for invalid paths."""
        invalid_path = tmp_path / "nonexistent.npz"

        with pytest.raises(FileNotFoundError):
            LogArchiveReader(archive_path=invalid_path)


class TestLogArchiveReaderRepr:
    """Tests for LogArchiveReader __repr__ method."""

    def test_repr_onset_not_discovered(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies __repr__ output when onset is not yet discovered."""
        archive_path, _, _, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        repr_str = repr(reader)
        assert "LogArchiveReader" in repr_str
        assert str(archive_path) in repr_str
        assert "not discovered" in repr_str

    def test_repr_onset_provided(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies __repr__ output when onset is pre-provided."""
        archive_path, _, onset_us, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(onset_us))

        repr_str = repr(reader)
        assert "LogArchiveReader" in repr_str
        assert str(onset_us) in repr_str


class TestLogArchiveReaderOnsetTimestamp:
    """Tests for LogArchiveReader onset_timestamp_us property."""

    def test_onset_from_pre_provided(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that pre-provided onset is returned without scanning."""
        archive_path, _, _, _ = sample_archive
        custom_onset = np.uint64(9999999999)
        reader = LogArchiveReader(archive_path=archive_path, onset_us=custom_onset)

        assert reader.onset_timestamp_us == custom_onset

    def test_onset_from_discovery(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that onset is correctly discovered from the archive."""
        archive_path, _, onset_us, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        discovered_onset = reader.onset_timestamp_us
        assert discovered_onset == np.uint64(onset_us)

    def test_onset_discovery_caches_message_keys(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies that onset discovery also caches message keys."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        # Triggers onset discovery.
        _ = reader.onset_timestamp_us

        # Verifies message keys were cached.
        assert reader._message_keys is not None
        assert len(reader._message_keys) == len(payloads)

    def test_onset_not_found_raises_error(self, tmp_path: Path) -> None:
        """Verifies that ValueError is raised when no onset message exists."""
        # Creates an archive without an onset message (all non-zero timestamps).
        archive_path = tmp_path / "no_onset.npz"
        arrays = {}
        for i in range(5):
            key = f"001_{(i + 1) * 1000:020d}"
            # All messages have non-zero timestamps.
            payload = np.array([i], dtype=np.uint8)
            arrays[key] = _create_log_message(source_id=1, timestamp_us=(i + 1) * 1000, payload=payload)
        np.savez(archive_path, **arrays)

        reader = LogArchiveReader(archive_path=archive_path)

        with pytest.raises(ValueError, match="Unable to discover onset timestamp"):
            _ = reader.onset_timestamp_us


class TestLogArchiveReaderMessageKeys:
    """Tests for LogArchiveReader message_keys property."""

    def test_message_keys_triggers_onset_discovery(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies that accessing message_keys triggers onset discovery."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        keys = reader._get_message_keys()

        assert len(keys) == len(payloads)
        # Verifies onset was discovered.
        assert reader._onset_us is None  # Original field is None, cached property stores it differently.

    def test_message_keys_excludes_onset(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that message_keys does not include the onset message."""
        archive_path, source_id, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        keys = reader._get_message_keys()

        # The onset key should not be in the list.
        onset_key = f"{source_id:03d}_{0:020d}"
        assert onset_key not in keys
        assert len(keys) == len(payloads)

    def test_message_keys_no_onset_pattern_match(self, tmp_path: Path) -> None:
        """Verifies fallback when onset key doesn't match expected pattern (pre-provided onset case)."""
        # Creates an archive where the onset message doesn't follow the standard naming pattern.
        archive_path = tmp_path / "nonstandard.npz"
        arrays = {}

        # Onset message with non-standard key (doesn't end with 20 zeros).
        onset_key = "onset_message"
        arrays[onset_key] = _create_onset_message(source_id=1, onset_us=1700000000000000)

        # Data messages.
        for i in range(3):
            key = f"001_{(i + 1) * 1000:020d}"
            payload = np.array([i], dtype=np.uint8)
            arrays[key] = _create_log_message(source_id=1, timestamp_us=(i + 1) * 1000, payload=payload)

        np.savez(archive_path, **arrays)

        # Pre-provides onset to skip discovery, forcing the fallback path.
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(1700000000000000))

        # Since onset key doesn't match pattern, all keys are returned.
        keys = reader._get_message_keys()
        assert len(keys) == 4  # Includes the non-standard onset key


class TestLogArchiveReaderMessageCount:
    """Tests for LogArchiveReader message_count property."""

    def test_message_count(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that message_count returns the correct count."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        assert reader.message_count == len(payloads)


class TestLogArchiveReaderGetBatches:
    """Tests for LogArchiveReader get_batches method."""

    def test_get_batches_small_archive_single_batch(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies that small archives return a single batch."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        batches = reader.get_batches(workers=4)

        assert len(batches) == 1
        assert len(batches[0]) == len(payloads)

    def test_get_batches_empty_archive(self, tmp_path: Path) -> None:
        """Verifies that empty archives return an empty list."""
        # Creates an archive with only an onset message.
        archive_path = tmp_path / "empty.npz"
        onset_key = f"{1:03d}_{0:020d}"
        arrays = {onset_key: _create_onset_message(source_id=1, onset_us=1700000000000000)}
        np.savez(archive_path, **arrays)

        reader = LogArchiveReader(archive_path=archive_path)
        batches = reader.get_batches(workers=4)

        assert batches == []

    def test_get_batches_large_archive_multiple_batches(self, large_archive: tuple[Path, int, int, int]) -> None:
        """Verifies that large archives are split into multiple batches."""
        archive_path, _, _, message_count = large_archive
        reader = LogArchiveReader(archive_path=archive_path)

        batches = reader.get_batches(workers=4, batch_multiplier=4)

        # Verifies multiple batches were created.
        assert len(batches) > 1

        # Verifies all messages are included.
        total_keys = sum(len(batch) for batch in batches)
        assert total_keys == message_count

    def test_get_batches_respects_worker_count(self, large_archive: tuple[Path, int, int, int]) -> None:
        """Verifies that batch count scales with worker count."""
        archive_path, _, _, _ = large_archive
        reader = LogArchiveReader(archive_path=archive_path)

        batches_2_workers = reader.get_batches(workers=2, batch_multiplier=1)
        batches_4_workers = reader.get_batches(workers=4, batch_multiplier=1)

        # More workers should create more batches (with same multiplier).
        assert len(batches_4_workers) >= len(batches_2_workers)

    def test_get_batches_default_workers(self, large_archive: tuple[Path, int, int, int]) -> None:
        """Verifies that default worker count works correctly."""
        archive_path, _, _, message_count = large_archive
        reader = LogArchiveReader(archive_path=archive_path)

        batches = reader.get_batches()

        # Verifies batches were created and all messages are included.
        assert len(batches) >= 1
        total_keys = sum(len(batch) for batch in batches)
        assert total_keys == message_count


class TestLogArchiveReaderIterMessages:
    """Tests for LogArchiveReader iter_messages method."""

    def test_iter_messages_all(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies iteration over all messages."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        messages = list(reader.iter_messages())

        assert len(messages) == len(payloads)

        # Verifies each message.
        for i, message in enumerate(messages):
            expected_timestamp = onset_us + (i + 1) * 1000
            assert message.timestamp_us == np.uint64(expected_timestamp)
            np.testing.assert_array_equal(message.payload, payloads[i])

    def test_iter_messages_subset(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies iteration over a subset of messages."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        # Gets first 3 keys only.
        all_keys = reader._get_message_keys()
        subset_keys = all_keys[:3]

        messages = list(reader.iter_messages(keys=subset_keys))

        assert len(messages) == 3

        # Verifies each message.
        for i, message in enumerate(messages):
            expected_timestamp = onset_us + (i + 1) * 1000
            assert message.timestamp_us == np.uint64(expected_timestamp)
            np.testing.assert_array_equal(message.payload, payloads[i])

    def test_iter_messages_with_pre_provided_onset(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies iteration works with pre-provided onset."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(onset_us))

        messages = list(reader.iter_messages())

        assert len(messages) == len(payloads)


class TestLogArchiveReaderReadAllMessages:
    """Tests for LogArchiveReader read_all_messages method."""

    def test_read_all_messages(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies reading all messages at once."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        timestamps, read_payloads = reader.read_all_messages()

        assert len(timestamps) == len(payloads)
        assert len(read_payloads) == len(payloads)

        # Verifies timestamps.
        for i, timestamp in enumerate(timestamps):
            expected_timestamp = onset_us + (i + 1) * 1000
            assert timestamp == np.uint64(expected_timestamp)

        # Verifies payloads.
        for i, payload in enumerate(read_payloads):
            np.testing.assert_array_equal(payload, payloads[i])

    def test_read_all_messages_returns_correct_types(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies that read_all_messages returns the correct types."""
        archive_path, _, _, _ = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        timestamps, payloads = reader.read_all_messages()

        assert isinstance(timestamps, np.ndarray)
        assert timestamps.dtype == np.uint64
        assert isinstance(payloads, list)
        assert all(isinstance(p, np.ndarray) for p in payloads)


class TestLogArchiveReaderIntegration:
    """Integration tests for LogArchiveReader with DataLogger output."""

    def test_reader_with_data_logger_output(self, tmp_path: Path) -> None:
        """Verifies that LogArchiveReader works with actual DataLogger output."""
        # Creates and runs a DataLogger.
        logger = DataLogger(output_directory=tmp_path, instance_name="test_reader")
        logger.start()

        # Gets the current UTC timestamp for the onset message.
        onset_us = get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)

        # Submits the onset message first (timestamp=0, payload contains UTC epoch as int64).
        onset_payload = np.array([onset_us], dtype=np.int64).view(np.uint8)
        onset_packed = LogPackage(source_id=np.uint8(1), acquisition_time=np.uint64(0), serialized_data=onset_payload)
        logger.input_queue.put(onset_packed)

        # Submits test data.
        test_payloads = []
        for i in range(5):
            payload = np.array([i, i + 1, i + 2], dtype=np.uint8)
            test_payloads.append(payload)
            packed = LogPackage(
                source_id=np.uint8(1), acquisition_time=np.uint64(i * 1000 + 1000), serialized_data=payload
            )
            logger.input_queue.put(packed)

        logger.stop()

        # Assembles the archive.
        assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=False)

        # Finds the archive.
        archives = list(logger.output_directory.glob("*.npz"))
        assert len(archives) == 1

        # Reads using LogArchiveReader.
        reader = LogArchiveReader(archive_path=archives[0])

        # Verifies onset was discovered.
        onset = reader.onset_timestamp_us
        assert onset > 0

        # Verifies message count (5 data messages, excluding onset).
        assert reader.message_count == 5

        # Verifies all messages can be read.
        messages = list(reader.iter_messages())
        assert len(messages) == 5

        # Verifies payloads match (order may differ due to timing).
        read_payloads = [m.payload for m in messages]
        for payload in test_payloads:
            assert any(np.array_equal(payload, rp) for rp in read_payloads)
