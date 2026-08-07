from types import MappingProxyType
from typing import Any, Self
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable as Callable

_YAML_EXCLUDE_METADATA_KEY: str
YAML_EXCLUDE_METADATA: MappingProxyType[str, bool]
_MAPPING_ARGUMENT_COUNT: int
_TYPE_HOOK_CACHE_SIZE: int
_LIBYAML_AVAILABLE: bool

def _serialize_value(value: Any) -> Any: ...
def _make_union_enum_hook(enum_types: list[type]) -> Callable[[Any], Any]: ...
def _make_mapping_key_hook(key_type: type) -> Callable[[Any], Any]: ...
def _collect_type_hooks(cls) -> dict[Any, Callable[[Any], Any]]: ...

@dataclass
class YamlConfig:
    def to_yaml(self, file_path: Path) -> None: ...
    @classmethod
    def restore_excluded_fields(cls, data: dict[Any, Any], file_path: Path) -> dict[Any, Any]: ...
    @classmethod
    def from_yaml(cls, file_path: Path) -> Self: ...
