"""Provides the two file writers the library offers, one for replacing a file another process reads and one for
creating a file nothing has opened yet.
"""

from __future__ import annotations

import os
import sys
from typing import IO, TYPE_CHECKING, Any
from contextlib import contextmanager

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import ensure_directory_exists

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Generator

_WRITE_PERMISSION_BITS: int = 0o666
"""The permission bits requested for a written file, before the process umask narrows them.

Notes:
    Matches the bits the built-in open() requests, so a file written this way carries the same permissions as every
    other file the library produces. The umask supplies the narrowing, which keeps the decision with the host rather
    than with this module.
"""

_RENAME_RETRY_COUNT: int = 5
"""The maximum number of times publishing a written file is attempted before the failure propagates."""

_RENAME_RETRY_DELAY_MILLISECONDS: int = 500
"""The delay in milliseconds before the second attempt to publish a written file.

Notes:
    The delay doubles on each further attempt, which suits a holder that keeps the destination open for a duration
    this module cannot predict. A scanner reading a small file releases it within the first delay, while one working
    through a large directory takes longer than a fixed delay would allow for.
"""


@contextmanager
def atomic_write(file_path: Path, *, binary: bool = False) -> Generator[IO[Any], None, None]:
    """Opens a temporary file that replaces the target path in one step once the caller finishes writing to it.

    Notes:
        Suits a destination that already exists and that another process may read, which covers a processing tracker,
        a configuration document, and a checksum. A file nothing has opened yet is written through ``direct_write()``
        instead, which costs one open() against the temporary file, the flush, and the rename this function pays.

        A reader of the target path observes either the previous file or the complete new one, never a partial write.
        Writing the destination directly instead truncates it first, so a writer killed mid-write leaves a truncated
        file where a complete one used to be.

        The temporary file is created in the destination's own directory, so the rename that publishes it stays within
        one filesystem and is therefore atomic. A destination whose parent directory does not exist yet has it created.

        The written file carries 0o644 on a host using the default 0o022 umask, since the temporary file requests the
        same bits the built-in open() requests and the umask narrows them. The permissions a destination carried
        before the write are NOT preserved, so a caller needing anything other than the default applies it with
        chmod() after this context exits.

        The contents reach the disk before the rename publishes them, so a host losing power immediately afterwards
        still finds the complete file. A caller writing a large number of small files pays that flush per file.

        A failure anywhere inside the context removes the temporary file and propagates, leaving the destination as it
        was.

    Args:
        file_path: The path to the file to write. The file is created when it does not exist and replaced when it does.
        binary: Determines whether the yielded file object accepts bytes instead of text.

    Yields:
        The file object to write the contents to.

    Raises:
        OSError: If the temporary file cannot be created or written.
        PermissionError: If the destination stays locked by another process for every publishing attempt.
    """
    ensure_directory_exists(path=file_path, is_file=True)

    # Names the temporary file after the destination and the writing process, so two processes writing the same
    # destination cannot collide on it, and a leftover file names the runtime that produced it.
    temporary_path = file_path.with_name(f".{file_path.name}.{os.getpid()}.tmp")

    # Opens the temporary file rather than drawing it from mkstemp(), which hardcodes the 0o600 bits that suit a
    # private scratch file and leave a published file readable only by the account that wrote it.
    descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _WRITE_PERMISSION_BITS)
    try:
        with os.fdopen(fd=descriptor, mode="wb" if binary else "w", encoding=None if binary else "utf-8") as file:
            yield file

            # Forces the data out of the userspace and kernel buffers before the rename publishes the file.
            file.flush()
            os.fsync(file.fileno())
        _publish_file(temporary_path=temporary_path, file_path=file_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def direct_write(file_path: Path, *, binary: bool = False) -> Generator[IO[Any], None, None]:
    """Opens the target path for writing, replacing whatever it already holds.

    Notes:
        Suits a file the caller creates rather than replaces, which covers every output written once into a fresh
        directory and every acquisition path writing a large number of files. A destination another process may read
        while the write runs is written through ``atomic_write()`` instead.

        Costs one open() and leaves the contents in the operating system's buffers, so a path writing a file per
        record pays neither the second open() and rename that publishing through a temporary file costs, nor the
        flush to disk that a durable write costs.

        A reader opening the path while this context is open observes a partially written file, and a writer killed
        partway leaves one behind. Both are acceptable for a file whose absence or truncation the caller detects
        anyway, and neither is acceptable for a file that already held a complete previous version.

        The written file carries 0o644 on a host using the default 0o022 umask, matching ``atomic_write()`` and the
        built-in open(). A destination whose parent directory does not exist yet has it created.

    Args:
        file_path: The path to the file to write. The file is created when it does not exist and truncated when it
            does.
        binary: Determines whether the yielded file object accepts bytes instead of text.

    Yields:
        The file object to write the contents to.

    Raises:
        OSError: If the file cannot be created or written.
    """
    ensure_directory_exists(path=file_path, is_file=True)

    with file_path.open(mode="wb" if binary else "w", encoding=None if binary else "utf-8") as file:
        yield file


def _publish_file(temporary_path: Path, file_path: Path) -> None:
    """Renames the written temporary file over the destination path.

    Notes:
        Windows refuses the rename while another process holds the destination open without sharing its deletion,
        which a scanner or an indexer does for as long as it takes to read the file. The attempt is repeated with a
        doubling delay there, since the holder releases the file on its own. Every other platform replaces an open
        destination without complaint and renames on the first attempt.

    Args:
        temporary_path: The path to the written temporary file.
        file_path: The destination path the temporary file is renamed to.

    Raises:
        PermissionError: If the destination stays locked by another process for every attempt.
    """
    if sys.platform != "win32":
        temporary_path.replace(target=file_path)
        return

    delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    delay = _RENAME_RETRY_DELAY_MILLISECONDS
    for attempt in range(_RENAME_RETRY_COUNT):
        try:
            temporary_path.replace(target=file_path)
        except PermissionError:
            if attempt == _RENAME_RETRY_COUNT - 1:
                raise
            delay_timer.delay(block=False, delay=delay, allow_sleep=True)
            delay *= 2
        else:
            return
