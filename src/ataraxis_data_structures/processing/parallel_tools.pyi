from contextlib import contextmanager
from collections.abc import Mapping, Generator

_RESIZABLE_THREAD_VARIABLES: tuple[str, ...]
_IMPORT_LATCHED_THREAD_VARIABLES: tuple[str, ...]

@contextmanager
def limit_worker_threads(
    thread_count: int = 1,
    maximum_thread_count: int | None = None,
    additional_thread_variables: Mapping[str, int] | None = None,
) -> Generator[None, None, None]: ...
def initialize_worker_threads(
    thread_count: int = 1,
    maximum_thread_count: int | None = None,
    additional_thread_variables: Mapping[str, int] | None = None,
) -> None: ...
def _resolve_thread_variables(
    thread_count: int,
    maximum_thread_count: int | None,
    additional_thread_variables: Mapping[str, int] | None,
    action: str,
) -> dict[str, str]: ...
