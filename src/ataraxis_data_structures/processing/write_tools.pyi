from typing import IO, Any
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Generator

_WRITE_PERMISSION_BITS: int
_RENAME_RETRY_COUNT: int
_RENAME_RETRY_DELAY_MILLISECONDS: int

@contextmanager
def atomic_write(file_path: Path, *, binary: bool = False) -> Generator[IO[Any], None, None]: ...
@contextmanager
def direct_write(file_path: Path, *, binary: bool = False) -> Generator[IO[Any], None, None]: ...
def _publish_file(temporary_path: Path, file_path: Path) -> None: ...
