"""Contains tests for classes and functions provided by the log_archive_reader.py module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp
from ataraxis_base_utilities import error_format, convert_scalar_to_bytes

from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    LOG_DIRECTORY_SUFFIX,
    DataLogger,
    LogMessage,
    LogPackage,
    LogArchiveReader,
    find_log_archive,
    assemble_log_archives,
    discover_log_archives,
    read_archive_message_count,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.fixture
def sample_archive(tmp_path: Path) -> tuple[Path, int, int, list[NDArray[np.uint8]]]:
    """Creates a sample archive with 10 messages for testing."""
    archive_path = tmp_path / "001_log.npz"
    source_id = 1
    onset_us = 1700000000000000  # Sample UTC epoch in microseconds.
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
    """Contains tests for the LogMessage dataclass."""

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
    """Contains tests for LogArchiveReader initialization."""

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
    """Contains tests for the LogArchiveReader __repr__() method."""

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
    """Contains tests for the LogArchiveReader onset_timestamp_us property."""

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

        _ = reader.onset_timestamp_us

        assert reader._message_keys is not None
        assert len(reader._message_keys) == len(payloads)

    def test_onset_not_found_raises_error(self, tmp_path: Path) -> None:
        """Verifies that ValueError is raised when no onset message exists."""
        # Creates an archive without an onset message (all non-zero timestamps).
        archive_path = tmp_path / "no_onset.npz"
        # All messages have non-zero timestamps.
        arrays = {
            f"001_{(index + 1) * 1000:020d}": _create_log_message(
                source_id=1, timestamp_us=(index + 1) * 1000, payload=np.array([index], dtype=np.uint8)
            )
            for index in range(5)
        }
        np.savez(file=archive_path, **arrays)

        reader = LogArchiveReader(archive_path=archive_path)

        with pytest.raises(ValueError, match="Unable to discover onset timestamp"):
            _ = reader.onset_timestamp_us


class TestLogArchiveReaderMessageKeys:
    """Contains tests for the LogArchiveReader message_keys property."""

    def test_message_keys_triggers_onset_discovery(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies that accessing message_keys triggers onset discovery."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        keys = reader._get_message_keys()

        assert len(keys) == len(payloads)
        # Original field is None, cached property stores it differently.
        assert reader._onset_us is None

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

        arrays.update(
            {
                f"001_{(index + 1) * 1000:020d}": _create_log_message(
                    source_id=1, timestamp_us=(index + 1) * 1000, payload=np.array([index], dtype=np.uint8)
                )
                for index in range(3)
            }
        )

        np.savez(file=archive_path, **arrays)

        # Pre-provides onset to skip discovery, forcing the fallback path.
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(1700000000000000))

        # Since onset key doesn't match pattern, all keys are returned.
        keys = reader._get_message_keys()
        assert len(keys) == 4  # Includes the non-standard onset key.


class TestLogArchiveReaderMessageCount:
    """Contains tests for the LogArchiveReader message_count property."""

    def test_message_count(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies that message_count returns the correct count."""
        archive_path, _, _, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        assert reader.message_count == len(payloads)


class TestLogArchiveReaderGetBatches:
    """Contains tests for the LogArchiveReader get_batches() method."""

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
        np.savez(file=archive_path, **arrays)

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
    """Contains tests for the LogArchiveReader iter_messages() method."""

    def test_iter_messages_all(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies iteration over all messages."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        messages = list(reader.iter_messages())

        assert len(messages) == len(payloads)

        for index, message in enumerate(messages):
            expected_timestamp = onset_us + (index + 1) * 1000
            assert message.timestamp_us == np.uint64(expected_timestamp)
            np.testing.assert_array_equal(message.payload, payloads[index])

    def test_iter_messages_subset(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies iteration over a subset of messages."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        # Gets first 3 keys only.
        all_keys = reader._get_message_keys()
        subset_keys = all_keys[:3]

        messages = list(reader.iter_messages(keys=subset_keys))

        assert len(messages) == 3

        for index, message in enumerate(messages):
            expected_timestamp = onset_us + (index + 1) * 1000
            assert message.timestamp_us == np.uint64(expected_timestamp)
            np.testing.assert_array_equal(message.payload, payloads[index])

    def test_iter_messages_with_pre_provided_onset(
        self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]
    ) -> None:
        """Verifies iteration works with pre-provided onset."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path, onset_us=np.uint64(onset_us))

        messages = list(reader.iter_messages())

        assert len(messages) == len(payloads)


class TestLogArchiveReaderReadAllMessages:
    """Contains tests for the LogArchiveReader read_all_messages() method."""

    def test_read_all_messages(self, sample_archive: tuple[Path, int, int, list[NDArray[np.uint8]]]) -> None:
        """Verifies reading all messages at once."""
        archive_path, _, onset_us, payloads = sample_archive
        reader = LogArchiveReader(archive_path=archive_path)

        timestamps, read_payloads = reader.read_all_messages()

        assert len(timestamps) == len(payloads)
        assert len(read_payloads) == len(payloads)

        for index, timestamp in enumerate(timestamps):
            expected_timestamp = onset_us + (index + 1) * 1000
            assert timestamp == np.uint64(expected_timestamp)

        for index, payload in enumerate(read_payloads):
            np.testing.assert_array_equal(payload, payloads[index])

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
        assert all(isinstance(payload, np.ndarray) for payload in payloads)


class TestLogArchiveReaderIntegration:
    """Contains integration tests for LogArchiveReader with DataLogger output."""

    def test_reader_with_data_logger_output(self, tmp_path: Path) -> None:
        """Verifies that LogArchiveReader works with actual DataLogger output."""
        # Creates and runs a DataLogger.
        logger = DataLogger(output_directory=tmp_path, instance_name="test_reader")
        logger.start()

        # Gets the current UTC timestamp for the onset message.
        onset_us = get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)

        # Submits the onset message first (timestamp=0, payload contains UTC epoch as uint64).
        onset_payload = convert_scalar_to_bytes(value=onset_us, dtype=np.dtype(np.uint64))
        onset_packed = LogPackage(source_id=np.uint8(1), acquisition_time=np.uint64(0), serialized_data=onset_payload)
        logger.input_queue.put(onset_packed)

        test_payloads = []
        for index in range(5):
            payload = np.array([index, index + 1, index + 2], dtype=np.uint8)
            test_payloads.append(payload)
            packed = LogPackage(
                source_id=np.uint8(1), acquisition_time=np.uint64(index * 1000 + 1000), serialized_data=payload
            )
            logger.input_queue.put(packed)

        logger.stop()

        assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, verbose=False)

        archives = list(logger.output_directory.glob("*.npz"))
        assert len(archives) == 1

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
        read_payloads = [message.payload for message in messages]
        for payload in test_payloads:
            assert any(np.array_equal(payload, read_payload) for read_payload in read_payloads)


def _create_log_message(source_id: int, timestamp_us: int, payload: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Creates a log message in the format expected by LogArchiveReader.

    Args:
        source_id: The source identifier (0-255).
        timestamp_us: The elapsed timestamp in microseconds.
        payload: The payload data.

    Returns:
        The serialized message bytes.
    """
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = convert_scalar_to_bytes(value=timestamp_us, dtype=np.dtype(np.uint64))
    return np.concatenate([source_bytes, timestamp_bytes, payload])


def _create_onset_message(source_id: int, onset_us: int) -> NDArray[np.uint8]:
    """Creates an onset message with timestamp=0 and the onset UTC epoch as payload.

    Args:
        source_id: The source identifier (0-255).
        onset_us: The UTC epoch onset timestamp in microseconds.

    Returns:
        The serialized onset message bytes.
    """
    source_bytes = np.array([source_id], dtype=np.uint8)
    timestamp_bytes = convert_scalar_to_bytes(value=0, dtype=np.dtype(np.uint64))
    onset_bytes = convert_scalar_to_bytes(value=onset_us, dtype=np.dtype(np.uint64))
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

    onset_key = f"{source_id:03d}_{0:020d}"
    arrays[onset_key] = _create_onset_message(source_id=source_id, onset_us=onset_us)

    payloads = []
    for index in range(message_count):
        elapsed_us = (index + 1) * 1000  # 1ms between messages
        payload = np.array([(index + payload_index) % 256 for payload_index in range(payload_size)], dtype=np.uint8)
        payloads.append(payload)

        message_key = f"{source_id:03d}_{elapsed_us:020d}"
        arrays[message_key] = _create_log_message(source_id=source_id, timestamp_us=elapsed_us, payload=payload)

    np.savez(file=archive_path, **arrays)

    return payloads


class TestArchiveDiscovery:
    """Contains tests for the module-level archive discovery and probing functions."""

    def test_find_log_archive_locates_a_nested_archive(self, tmp_path: Path) -> None:
        """Verifies that the search descends into subdirectories to resolve an archive from its source ID."""
        nested = tmp_path / "session" / "raw_data" / "behavior_data_log"
        nested.mkdir(parents=True)
        archive_path = nested / f"7{LOG_ARCHIVE_SUFFIX}"
        _create_test_archive(archive_path=archive_path, source_id=7, onset_us=1700000000000000, message_count=3)

        assert find_log_archive(log_directory=tmp_path, source_id="7") == archive_path

    def test_find_log_archive_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        """Verifies that a log directory that does not exist is rejected."""
        missing = tmp_path / "never_created"
        message = (
            f"Unable to find the log archive of source '7' in '{missing}'. The path does not exist or is not a "
            f"directory."
        )
        with pytest.raises(FileNotFoundError, match=error_format(message)):
            find_log_archive(log_directory=missing, source_id="7")

    def test_find_log_archive_rejects_a_directory_holding_no_matching_archive(self, tmp_path: Path) -> None:
        """Verifies that a tree holding no archive for the requested source is rejected."""
        message = (
            f"Unable to find the log archive of source '7' in '{tmp_path}'. No file named '7{LOG_ARCHIVE_SUFFIX}' "
            f"was found anywhere under the directory."
        )
        with pytest.raises(FileNotFoundError, match=error_format(message)):
            find_log_archive(log_directory=tmp_path, source_id="7")

    def test_find_log_archive_rejects_an_ambiguous_source(self, tmp_path: Path) -> None:
        """Verifies that a tree holding one archive per logger for the same source ID is rejected as ambiguous."""
        for logger_name in ("first_data_log", "second_data_log"):
            directory = tmp_path / logger_name
            directory.mkdir()
            _create_test_archive(
                archive_path=directory / f"7{LOG_ARCHIVE_SUFFIX}",
                source_id=7,
                onset_us=1700000000000000,
                message_count=1,
            )

        with pytest.raises(ValueError, match="but 2 were found"):
            find_log_archive(log_directory=tmp_path, source_id="7")

    def test_discover_log_archives_maps_every_archive_to_its_source(self, tmp_path: Path) -> None:
        """Verifies that discovery keys each archive stored directly in the directory by its source ID."""
        for source_id in (3, 11):
            _create_test_archive(
                archive_path=tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
                source_id=source_id,
                onset_us=1700000000000000,
                message_count=2,
            )
        nested = tmp_path / "nested"
        nested.mkdir()
        _create_test_archive(
            archive_path=nested / f"5{LOG_ARCHIVE_SUFFIX}", source_id=5, onset_us=1700000000000000, message_count=2
        )

        discovered = discover_log_archives(log_directory=tmp_path)

        # The nested archive belongs to a different logger, so it is absent from this logger's mapping.
        assert discovered == {
            "3": tmp_path / f"3{LOG_ARCHIVE_SUFFIX}",
            "11": tmp_path / f"11{LOG_ARCHIVE_SUFFIX}",
        }

    def test_discover_log_archives_omits_a_directory_carrying_an_archive_name(self, tmp_path: Path) -> None:
        """Verifies that a directory whose name matches the archive pattern contributes no entry."""
        (tmp_path / f"7{LOG_ARCHIVE_SUFFIX}").mkdir()
        _create_test_archive(
            archive_path=tmp_path / f"3{LOG_ARCHIVE_SUFFIX}",
            source_id=3,
            onset_us=1700000000000000,
            message_count=1,
        )

        assert discover_log_archives(log_directory=tmp_path) == {"3": tmp_path / f"3{LOG_ARCHIVE_SUFFIX}"}

    def test_discover_log_archives_omits_a_file_naming_no_source(self, tmp_path: Path) -> None:
        """Verifies that a file whose whole name is the archive suffix contributes no entry."""
        (tmp_path / LOG_ARCHIVE_SUFFIX).write_bytes(b"")

        assert discover_log_archives(log_directory=tmp_path) == {}

    def test_discover_log_archives_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        """Verifies that a log directory that does not exist is rejected."""
        missing = tmp_path / "never_created"
        message = (
            f"Unable to discover the log archives stored in '{missing}'. The path does not exist or is not a directory."
        )
        with pytest.raises(FileNotFoundError, match=error_format(message)):
            discover_log_archives(log_directory=missing)

    @pytest.mark.parametrize("message_count", [0, 1, 25])
    def test_read_archive_message_count_matches_the_reader(self, tmp_path: Path, message_count: int) -> None:
        """Verifies that the probe agrees with the count the reader derives from the same archive."""
        archive_path = tmp_path / f"9{LOG_ARCHIVE_SUFFIX}"
        _create_test_archive(
            archive_path=archive_path, source_id=9, onset_us=1700000000000000, message_count=message_count
        )

        assert read_archive_message_count(archive_path=archive_path) == message_count
        assert (
            read_archive_message_count(archive_path=archive_path)
            == LogArchiveReader(archive_path=archive_path).message_count
        )

    def test_read_archive_message_count_rejects_a_missing_archive(self, tmp_path: Path) -> None:
        """Verifies that an archive path that does not resolve to a file is rejected."""
        missing = tmp_path / "absent_log.npz"
        message = (
            f"Unable to read the message count of the log archive at '{missing}'. The path does not exist or is not "
            f"a file."
        )
        with pytest.raises(FileNotFoundError, match=error_format(message)):
            read_archive_message_count(archive_path=missing)

    def test_data_logger_output_uses_the_exported_naming_constants(self, tmp_path: Path) -> None:
        """Verifies that a logger names its output directory and its archives with the exported suffixes."""
        logger = DataLogger(output_directory=tmp_path, instance_name="behavior")
        output_directory = tmp_path / f"behavior{LOG_DIRECTORY_SUFFIX}"
        assert output_directory.is_dir()

        logger.start()
        logger.input_queue.put(
            LogPackage(
                source_id=np.uint8(4),
                acquisition_time=np.uint64(0),
                serialized_data=convert_scalar_to_bytes(value=1700000000000000, dtype=np.dtype(np.uint64)),
            )
        )
        logger.input_queue.put(
            LogPackage(
                source_id=np.uint8(4),
                acquisition_time=np.uint64(1000),
                serialized_data=np.array([1, 2], dtype=np.uint8),
            )
        )
        logger.stop()
        assemble_log_archives(
            log_directory=output_directory, remove_sources=True, verify_integrity=False, verbose=False
        )

        archives = discover_log_archives(log_directory=output_directory)

        assert archives == {"4": output_directory / f"4{LOG_ARCHIVE_SUFFIX}"}
        assert read_archive_message_count(archive_path=archives["4"]) == 1
