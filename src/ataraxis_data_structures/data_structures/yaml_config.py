"""Provides the YamlConfig class, which extends the standard Python 'dataclass' class with methods to cache and retrieve
its data from a .yml (YAML) file.
"""

from enum import Enum
from types import UnionType
from typing import Any, Self, Union, get_args, get_origin, get_type_hints
from pathlib import Path
from dataclasses import fields, dataclass, is_dataclass
from collections.abc import Callable

import yaml
from dacite import Config, from_dict
from ataraxis_base_utilities import console, ensure_directory_exists


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

    # Checks Path before str, since PurePosixPath / PureWindowsPath are not str subclasses, but PosixPath and
    # WindowsPath inherit from os.PathLike, not str.
    if isinstance(value, Path):
        return str(value)

    # Checks Enum before str/int, since StrEnum and IntEnum subclass str and int respectively.
    if isinstance(value, Enum):
        return value.value

    # Recursively serializes dataclass instances into dicts.
    if is_dataclass(value) and not isinstance(value, type):
        return {data_field.name: _serialize_value(getattr(value, data_field.name)) for data_field in fields(value)}

    if isinstance(value, dict):
        return {_serialize_value(k): _serialize_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    # Passes through str, int, float, bool, and other YAML-native scalars unchanged.
    return value


def _make_union_enum_hook(enum_types: list[type]) -> Callable[[Any], Any]:
    """Creates a dacite type hook for union annotations that contain Enum subclasses.

    Notes:
        The returned hook tries each enum constructor in order. If a constructor succeeds, the enum member is returned.
        If none succeed (e.g., the value is not a valid member of any listed enum), the raw value passes through
        unchanged. This ensures correct deserialization regardless of annotation order (``str | Color`` works
        identically to ``Color | str``) because dacite invokes the union-level hook before iterating individual union
        members.

        This approach relies on Enum constructors raising ``ValueError`` for invalid members, which provides a natural
        discriminator. It is intentionally not used for Path unions because ``Path(any_string)`` always succeeds,
        making discrimination impossible.

    Args:
        enum_types: The Enum subclass types to try converting to, in the order they should be attempted.

    Returns:
        A callable that accepts a raw YAML value and returns the corresponding Enum member if valid, or the raw value
        unchanged if no enum constructor succeeds.
    """
    targets = list(enum_types)

    def hook(value: Any) -> Any:
        """Attempts to convert the value to one of the target Enum types, falling back to the raw value."""
        if value is None:
            return None
        for enum_class in targets:
            try:
                return enum_class(value)
            except (ValueError, KeyError):
                continue
        return value

    return hook


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
                    if not isinstance(argument, type):
                        continue
                    try:
                        if issubclass(argument, Enum) and argument is not Enum:
                            enum_targets.append(argument)
                    except TypeError:
                        pass
                if enum_targets:
                    hooks[type_hint] = _make_union_enum_hook(enum_types=enum_targets)

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
        except TypeError:
            return

        # Registers Enum subclasses (but not the abstract Enum base itself).
        try:
            if issubclass(type_hint, Enum) and type_hint is not Enum:
                hooks[type_hint] = type_hint
                return
        except TypeError:
            return

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
        except (TypeError, NameError, AttributeError):
            return

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
            YAML files human-readable while preserving full type fidelity on round-trip via ``from_yaml()``.

        Args:
            file_path: The path to the .yaml file to write.

        Raises:
            ValueError: If the file_path does not point to a file with a '.yaml' or '.yml' extension.
        """
        # Defines YAML formatting options. The purpose of these settings is to make YAML blocks more readable when
        # being edited by the user.
        yaml_formatting = {
            "default_style": "",  # Use double quotes for scalars as needed
            "default_flow_style": False,  # Use block style for mappings
            "indent": 10,  # The number of spaces for indentation
            "width": 200,  # Maximum line width before wrapping
            "explicit_start": True,  # Mark the beginning of the document with ___
            "explicit_end": True,  # Mark the end of the document with ___
            "sort_keys": False,  # Preserves the order of the keys as written by creators
        }

        # Ensures that the output file path points to a .yaml (or .yml) file.
        if file_path.suffix not in {".yaml", ".yml"}:
            message: str = (
                f"Invalid file path provided when attempting to write the dataclass instance to a .yaml file. "
                f"Expected a path ending in the '.yaml' or '.yml' extension as 'file_path' argument, but encountered "
                f"{file_path}."
            )
            console.error(message=message, error=ValueError)

        # If necessary, creates the missing directory components of the file_path.
        ensure_directory_exists(file_path)

        # Serializes the dataclass to a YAML-safe dict tree (Path -> str, Enum -> value, tuple -> list) and writes it
        # to the .yaml file.
        with file_path.open("w") as yaml_file:
            yaml.dump(data=_serialize_value(self), stream=yaml_file, **yaml_formatting)  # type: ignore[call-overload]

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
            ValueError: If the provided file path does not point to a .yaml or .yml file.
        """
        # Ensures that file_path points to a .yaml / .yml file.
        if file_path.suffix not in {".yaml", ".yml"}:
            message: str = (
                f"Invalid file path provided when attempting to create the dataclass instance using the data from a "
                f".yaml file. Expected the path ending in the '.yaml' or '.yml' extension as 'file_path' argument, but "
                f"encountered {file_path}."
            )
            console.error(message=message, error=ValueError)

        # Builds type_hooks from the class hierarchy to auto-convert str -> Path, raw value -> Enum, etc. The cast
        # list converts YAML lists back to tuples at the field level. check_types=False is preserved for backward
        # compatibility with union annotations like ``BaselineMethod | str``.
        type_hooks = _collect_type_hooks(cls)
        class_config = Config(type_hooks=type_hooks, cast=[tuple], check_types=False)

        # Loads the data from the .yaml file.
        with file_path.open() as yml_file:
            data = yaml.safe_load(yml_file)

        # Converts the imported data to a Python dictionary.
        data_dictionary: dict[Any, Any] = dict(data)

        # Uses dacite to instantiate the class using the imported dictionary.
        # noinspection PyTypeChecker
        return from_dict(data_class=cls, data=data_dictionary, config=class_config)
