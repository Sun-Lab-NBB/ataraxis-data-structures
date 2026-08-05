"""Provides the YamlConfig class, which extends the standard Python 'dataclass' class with methods to cache and retrieve
its data from a .yaml (YAML) file.
"""

import os
from enum import Enum
from types import UnionType, MappingProxyType
from typing import Any, Self, Union, get_args, get_origin, get_type_hints
from pathlib import Path
from tempfile import mkstemp
from dataclasses import fields, dataclass, is_dataclass
from collections.abc import Mapping, Callable

import yaml
from dacite import Config, from_dict
from ataraxis_base_utilities import console, ensure_directory_exists

_YAML_EXCLUDE_METADATA_KEY: str = "yaml_exclude"
"""The dataclass field metadata key that keeps a field out of the serialized document."""

YAML_EXCLUDE_METADATA: MappingProxyType[str, bool] = MappingProxyType({_YAML_EXCLUDE_METADATA_KEY: True})
"""The dataclass field metadata that keeps a field out of the serialized document.

Notes:
    A field declared as ``field(metadata=YAML_EXCLUDE_METADATA)`` is skipped when the instance is written, which suits
    a field describing where the instance lives rather than what it holds. A class excluding a field that its
    constructor requires supplies the value back through ``restore_excluded_fields()``.
"""

_MAPPING_ARGUMENT_COUNT: int = 2
"""The number of type arguments a mapping annotation carries, which is its key type followed by its value type."""

_LIBYAML_AVAILABLE: bool = hasattr(yaml, "CSafeLoader")
"""Determines whether the PyYAML build provides the libyaml-backed parser and emitter. Reading and writing both
select them wherever they are available, since they carry values through the same safe constructor and the same
representer as the pure-Python implementations while running about an order of magnitude faster. Builds without
libyaml fall back to the pure-Python implementations."""


def _serialize_value(value: Any) -> Any:
    """Recursively converts a dataclass instance or any nested value into a YAML-safe dictionary tree.

    Operates on a fresh dict tree and never mutates the original instance.

    Args:
        value: The value to serialize. Dataclass instances, Path objects, Enum members, dicts, lists, and tuples are
            recursively converted. YAML-native scalars (str, int, float, bool) pass through unchanged.

    Returns:
        A YAML-safe representation of the input value: Path instances become strings, Enum members become their raw
        values, tuples become lists, and dataclass instances become dicts.
    """
    if value is None:
        return None

    # Checks Path before the scalar pass-through, since Path instances are not str subclasses and require explicit
    # conversion to a serializable string.
    if isinstance(value, Path):
        return value.as_posix()

    # Checks Enum before str/int, since StrEnum and IntEnum subclass str and int respectively.
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return {
            data_field.name: _serialize_value(value=getattr(value, data_field.name))
            for data_field in fields(value)
            if not data_field.metadata.get(_YAML_EXCLUDE_METADATA_KEY, False)
        }

    if isinstance(value, dict):
        return {
            _serialize_value(value=dict_key): _serialize_value(value=dict_value)
            for dict_key, dict_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_serialize_value(value=item) for item in value]

    return value


def _make_union_enum_hook(enum_types: list[type]) -> Callable[[Any], Any]:
    """Creates a dacite type hook for union annotations that contain Enum subclasses.

    Notes:
        The returned hook tries each enum constructor in order. If a constructor succeeds, the enum member is returned.
        If none succeed (e.g., the value is not a valid member of any listed enum), the raw value passes through
        unchanged. This ensures correct deserialization regardless of annotation order (``str | Color`` works
        identically to ``Color | str``) because dacite invokes the union-level hook before iterating individual union
        members.

        This approach relies on Enum constructors raising ``ValueError`` or ``KeyError`` for invalid members, which
        provides a natural discriminator. It is intentionally not used for Path unions because ``Path(any_string)``
        always succeeds, making discrimination impossible.

    Args:
        enum_types: The Enum subclass types to try converting to, in the order they should be attempted.

    Returns:
        A callable that accepts a raw YAML value and returns the corresponding Enum member if valid, or the raw value
        unchanged if no enum constructor succeeds.
    """
    targets = list(enum_types)

    def _hook(value: Any) -> Any:
        """Attempts to convert the value to one of the target Enum types, falling back to the raw value."""
        if value is None:
            return None
        for enum_class in targets:
            try:
                return enum_class(value)
            except (ValueError, KeyError):
                continue
        return value

    return _hook


def _make_mapping_key_hook(key_type: type) -> Callable[[Any], Any]:
    """Creates a dacite type hook that converts the keys of a mapping to the type its annotation names.

    Notes:
        dacite converts a mapping's values through the hook registered for the value type and hands every key through
        untouched, so a field annotated with a Path or Enum key would otherwise load with the raw strings YAML stores.
        Registering this hook against the mapping annotation itself fires it before dacite descends into the values,
        which mirrors the key conversion ``_serialize_value`` performs on the way out.

    Args:
        key_type: The type each key is converted to.

    Returns:
        A callable that accepts a raw YAML mapping and returns the same mapping with its keys converted.
    """

    def _hook(value: Any) -> Any:
        """Converts every key of the mapping, leaving the values for dacite to convert."""
        if not isinstance(value, dict):
            return value
        return {key_type(key): item for key, item in value.items()}

    return _hook


def _collect_type_hooks(cls: type) -> dict[Any, Callable[[Any], Any]]:
    """Builds a dacite ``type_hooks`` dictionary by introspecting the dataclass class hierarchy.

    Discovers all Path and Enum subclass types referenced in field annotations (including inside generics and nested
    dataclasses) and returns a mapping that dacite uses to call the appropriate constructor on raw YAML values during
    deserialization.

    Notes:
        For union annotations containing Enum subclasses (e.g., ``str | Color`` or ``int | Priority``), a union-level
        hook is registered that tries each enum constructor before dacite's default left-to-right member iteration.
        This makes deserialization order-independent: ``str | Color`` and ``Color | str`` both correctly produce enum
        members for valid values, and fall back to the primitive type for non-member values.

    Args:
        cls: The dataclass type to introspect. All field annotations in the class and any nested dataclasses are
            walked to discover Path and Enum subclass types.

    Returns:
        A dictionary mapping types (concrete or union) to callables that dacite uses as type hooks during
        deserialization. Concrete Path and Enum types map to their own constructors, while union types containing
        Enum subclasses map to discriminating hook functions.
    """
    hooks: dict[Any, Callable[[Any], Any]] = {}
    visited: set[type] = set()

    def _walk_type(type_hint: Any) -> None:
        """Recursively walks a type annotation, registering hooks for Path and Enum subclasses.

        Args:
            type_hint: The type annotation to process. Can be a concrete type, a generic alias (e.g., ``list[Path]``),
                or a union type (e.g., ``str | Color``).
        """
        type_arguments = get_args(type_hint)
        if type_arguments:
            # Registers a union-level hook if the union contains any Enum subclass members. This fires before dacite
            # iterates union members, making annotation order irrelevant for enum conversion.
            if isinstance(type_hint, UnionType) or get_origin(type_hint) is Union:
                enum_targets: list[type] = []
                for argument in type_arguments:
                    if not isinstance(argument, type):  # pragma: no cover
                        continue  # pragma: no cover
                    try:
                        if issubclass(argument, Enum) and argument is not Enum:
                            enum_targets.append(argument)
                    except TypeError:  # pragma: no cover
                        pass  # pragma: no cover
                if enum_targets:
                    hooks[type_hint] = _make_union_enum_hook(enum_types=enum_targets)

            # Registers a key-converting hook for a mapping whose key annotation names a Path or Enum type, since
            # dacite converts mapping values alone.
            mapping_origin = get_origin(type_hint)
            if (
                isinstance(mapping_origin, type)
                and issubclass(mapping_origin, Mapping)
                and len(type_arguments) == _MAPPING_ARGUMENT_COUNT
            ):
                key_type = type_arguments[0]
                if isinstance(key_type, type) and (
                    issubclass(key_type, Path) or (issubclass(key_type, Enum) and key_type is not Enum)
                ):
                    hooks[type_hint] = _make_mapping_key_hook(key_type=key_type)

            # Recurses into all generic arguments (union members, list items, dict values, etc.).
            for argument in type_arguments:
                _walk_type(type_hint=argument)
            return

        # Only processes concrete types from here.
        if not isinstance(type_hint, type):
            return

        if type_hint in visited:
            return
        visited.add(type_hint)

        # Registers Path subclasses. dacite calls Path(str_value) during deserialization.
        try:
            if issubclass(type_hint, Path):
                hooks[type_hint] = type_hint
                return
        except TypeError:  # pragma: no cover
            return  # pragma: no cover

        # Registers Enum subclasses (but not the abstract Enum base itself).
        try:
            if issubclass(type_hint, Enum) and type_hint is not Enum:
                hooks[type_hint] = type_hint
                return
        except TypeError:  # pragma: no cover
            return  # pragma: no cover

        # Recurses into nested dataclass annotations to discover their Path/Enum fields.
        if is_dataclass(type_hint):
            _walk_dataclass(dataclass_type=type_hint)

    def _walk_dataclass(dataclass_type: type) -> None:
        """Introspects a dataclass's type hints and walks each annotation.

        Args:
            dataclass_type: The dataclass type whose field annotations should be walked.
        """
        try:
            hints = get_type_hints(dataclass_type)
        except (TypeError, NameError, AttributeError):  # pragma: no cover
            return  # pragma: no cover

        for hint_type in hints.values():
            _walk_type(type_hint=hint_type)

    _walk_dataclass(dataclass_type=cls)
    return hooks


@dataclass
class YamlConfig:
    """Extends the standard Python dataclass with methods to save and load its data from a .yaml (YAML) file.

    Notes:
        This class is designed to be subclassed by custom dataclasses so that they inherit the YAML saving and loading
        functionality. Serialization automatically converts Path instances to strings, Enum members to their raw values,
        and tuples to lists. Deserialization reverses these conversions based on the dataclass's type annotations.
    """

    def to_yaml(self, file_path: Path) -> None:
        """Saves the instance's data as the specified .yaml (YAML) file.

        Notes:
            Path fields are serialized as strings, Enum fields as their raw values, and tuples as lists. This keeps
            YAML files human-readable while preserving type fidelity on round-trip via ``from_yaml()`` for concretely
            annotated fields. A field whose annotation unions ``Path`` with ``str`` cannot be discriminated on load
            and is restored as a string.

            The file is written through a temporary file and renamed into place, so a process killed mid-write leaves
            the previously saved file intact. Both reading and writing use UTF-8 regardless of the host locale.

        Args:
            file_path: The path to the .yaml file to write.

        Raises:
            ValueError: If ``file_path`` does not point to a file with a '.yaml' or '.yml' extension.
        """
        # Defines YAML formatting options that keep YAML blocks readable when edited by the user.
        yaml_formatting = {
            # Uses plain (unquoted) scalar style, quoting only when required.
            "default_style": "",
            # Uses block style for mappings.
            "default_flow_style": False,
            "indent": 10,
            "width": 200,
            # Marks the beginning of the document with the "---" prefix.
            "explicit_start": True,
            # Marks the end of the document with the "..." suffix.
            "explicit_end": True,
            # Preserves the key order as written by the dataclass authors.
            "sort_keys": False,
        }

        # Ensures that the output file path points to a .yaml (or .yml) file.
        if file_path.suffix not in {".yaml", ".yml"}:
            message: str = (
                f"Unable to write the dataclass instance to a .yaml file using the provided file path. The "
                f"'file_path' argument must end in the '.yaml' or '.yml' extension, but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # If necessary, creates the missing directory components of the file_path. The guard above accepts a .yaml or
        # .yml path alone, so the path is always a file path and the directory to create is its parent.
        ensure_directory_exists(path=file_path, is_file=True)

        # Serializes the dataclass to a YAML-safe dict tree (Path -> str, Enum -> value, tuple -> list) and writes it
        # through a temporary file created in the destination's own directory, so the rename below stays inside one
        # filesystem and is therefore atomic. A writer killed mid-dump leaves the previous complete document in place,
        # rather than the empty file that truncating the destination first would leave.
        descriptor, temporary_path = mkstemp(dir=file_path.parent, prefix=f".{file_path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd=descriptor, mode="w", encoding="utf-8") as yaml_file:
                yaml.dump(  # type: ignore[call-overload]
                    data=_serialize_value(value=self),
                    stream=yaml_file,
                    Dumper=yaml.CDumper if _LIBYAML_AVAILABLE else yaml.Dumper,
                    **yaml_formatting,
                )
                # Forces the data out of the userspace and kernel buffers before the rename publishes the file.
                yaml_file.flush()
                os.fsync(yaml_file.fileno())
            Path(temporary_path).replace(target=file_path)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise

    @classmethod
    def restore_excluded_fields(cls, data: dict[Any, Any], file_path: Path) -> dict[Any, Any]:  # noqa: ARG003
        """Returns the loaded mapping extended with the values of any field this class excludes from serialization.

        Notes:
            This method exists to be replaced by subclasses. The implementation here excludes no field and returns
            the mapping untouched, which is correct for every class that serializes all of its fields.

            A subclass marking a constructor-required field with ``YAML_EXCLUDE_METADATA`` overrides this method
            to supply that field's value. The written document carries no entry for such a field, so deserialization
            cannot build the instance without one. The path the document was read from is passed in, because a field
            excluded this way usually records where the instance lives.

            ``from_yaml()`` calls this method between reading the document and building the instance, so it belongs
            to the deserialization machinery rather than to the API a caller invokes.

        Args:
            data: The top-level mapping read from the .yaml file.
            file_path: The path the mapping was read from.

        Returns:
            The mapping to build the instance from, which is the input unchanged when no field is excluded.
        """
        return data

    @classmethod
    def from_yaml(cls, file_path: Path) -> Self:
        """Instantiates the class using the data loaded from the provided .yaml (YAML) file.

        Notes:
            Deserialization automatically converts YAML-native types back to the annotated Python types: strings
            to Path instances, raw values to Enum members, and lists to tuples where applicable. Type hooks are
            derived from the dataclass's field annotations, so no manual conversion boilerplate is needed in
            subclasses.

        Args:
            file_path: The path to the .yaml file that stores the instance's data.

        Returns:
            A new class instance that stores the data read from the .yaml file.

        Raises:
            ValueError: If the provided file path does not point to a .yaml or .yml file, or if the file does not
                contain a top-level mapping.
            FileNotFoundError: If no file exists at the provided file path.
        """
        # Ensures that file_path points to a .yaml / .yml file.
        if file_path.suffix not in {".yaml", ".yml"}:
            message: str = (
                f"Unable to create the dataclass instance using the data from a .yaml file. The 'file_path' argument "
                f"must end in the '.yaml' or '.yml' extension, but got {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Builds type_hooks from the class hierarchy to auto-convert str -> Path, raw value -> Enum, etc. The cast
        # list converts YAML lists back to tuples at the field level. check_types=False allows union annotations
        # such as ``Enum | str`` to accept either member.
        type_hooks = _collect_type_hooks(cls=cls)
        class_config = Config(type_hooks=type_hooks, cast=[tuple], check_types=False)

        # Loads the data from the .yaml file. Both parsers build values through the same safe constructor, so they
        # differ in speed alone. Each loader is named literally, since a loader resolved through a variable reads as
        # an arbitrary-object deserialization risk to static analysis.
        with file_path.open(encoding="utf-8") as yaml_file:
            data = (
                yaml.load(stream=yaml_file, Loader=yaml.CSafeLoader)
                if _LIBYAML_AVAILABLE
                else yaml.safe_load(stream=yaml_file)
            )

        # Ensures the loaded data is a top-level mapping. An empty file yields None, and a scalar or sequence document
        # yields a non-mapping type that cannot seed a dataclass instance.
        if not isinstance(data, dict):
            message = (
                f"Unable to create the dataclass instance using the data from the {file_path} .yaml file. The file "
                f"must contain a top-level mapping, but got {type(data).__name__}."
            )
            console.error(message=message, error=ValueError)

        data_dictionary: dict[Any, Any] = cls.restore_excluded_fields(data=dict(data), file_path=file_path)

        return from_dict(data_class=cls, data=data_dictionary, config=class_config)
