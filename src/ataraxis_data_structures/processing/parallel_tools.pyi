from contextlib import contextmanager
from collections.abc import Generator

_THREAD_LIMIT_VARIABLES: tuple[str, ...]

@contextmanager
def limit_worker_threads(thread_count: int = 1) -> Generator[None, None, None]: ...
def initialize_worker_threads(thread_count: int = 1) -> None: ...
