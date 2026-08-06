"""Contains tests for classes and methods provided by the yaml_config.py module."""

from enum import IntEnum, StrEnum
from typing import Any, Union, Optional
from pathlib import Path
from dataclasses import field, dataclass

import yaml
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import YAML_EXCLUDE_METADATA, YamlConfig
from ataraxis_data_structures.data_structures.yaml_config import _serialize_value, _collect_type_hooks


class Color(StrEnum):
    """Defines the test StrEnum for type-aware round-trip tests."""

    RED = "red"
    """Represents the red color."""
    GREEN = "green"
    """Represents the green color."""
    BLUE = "blue"
    """Represents the blue color."""


class Priority(IntEnum):
    """Defines the test IntEnum for type-aware round-trip tests."""

    LOW = 1
    """Represents low priority."""
    MEDIUM = 2
    """Represents medium priority."""
    HIGH = 3
    """Represents high priority."""


@pytest.mark.parametrize(
    "config_path, expected_content",
    [
        (Path("config1.yaml"), {"string_value": "value1", "integer_value": 2, "nested": {}, "list": []}),
        (Path("config2.yml"), {"string_value": "", "integer_value": 0, "nested": {"key": "value"}, "list": [1, 2, 3]}),
        (Path("empty_config.yaml"), {"string_value": "", "integer_value": 0, "nested": {}, "list": []}),
    ],
)
def test_yaml_config_to_yaml(tmp_path: Path, config_path: Path, expected_content: dict[str, Any]) -> None:
    """Verifies the functionality of the YamlConfig class to_yaml() method.

    Verifies saving a simple key-value pair configuration to a .yaml file, a nested configuration with lists to a .yml
    file, and an empty configuration to a .yaml file.
    """

    @dataclass
    class TestConfig(YamlConfig):
        string_value: str = ""
        integer_value: int = 0
        nested: dict = field(default_factory=dict)
        list: list = field(default_factory=list)

    # Generates and dumps the config as a .yaml file.
    config = TestConfig(**expected_content)
    full_path = tmp_path.joinpath(config_path)
    config.to_yaml(file_path=full_path)

    # Verifies that the file was created and contains data.
    assert full_path.exists()
    assert full_path.stat().st_size > 0, f"File {full_path} is empty"

    # Manually reads and verifies the config data.
    with full_path.open("r") as yaml_file:
        loaded_content = yaml.safe_load(yaml_file)
        assert loaded_content == expected_content, f"Expected {expected_content}, but got {loaded_content}"


def test_yaml_config_to_yaml_errors(tmp_path: Path) -> None:
    """Verifies the error-handling behavior of the YamlConfig class to_yaml() method."""

    @dataclass
    class TestConfig(YamlConfig):
        pass

    config = TestConfig()
    invalid_path = tmp_path / "invalid.txt"

    error_message: str = (
        f"Unable to write the dataclass instance to a .yaml file using the provided file path. The "
        f"'file_path' argument must end in the '.yaml' or '.yml' extension, but got {invalid_path}."
    )

    with pytest.raises(ValueError, match=error_format(error_message)):
        config.to_yaml(file_path=invalid_path)


@pytest.mark.parametrize(
    "config_path, content",
    [
        (Path("config1.yaml"), {"string_value": "value1", "integer_value": 2, "nested": None, "list": None}),
        (Path("config2.yml"), {"nested": {"key": "value"}, "list": [1, 2, 3]}),
    ],
)
def test_yaml_config_from_yaml(tmp_path: Path, config_path: Path, content: dict[str, Any]) -> None:
    """Verifies the functionality of the YamlConfig class from_yaml() method.

    Verifies loading a simple key-value pair configuration from a .yaml file and a nested configuration with lists from
    a .yml file.
    """

    @dataclass
    class TestConfig(YamlConfig):
        string_value: str = ""
        integer_value: int = 0
        nested: dict | None = None
        list: Optional[list] = None  # noqa: UP045 - field name shadows builtin, X | None fails

    full_path = tmp_path / config_path
    with full_path.open("w") as yaml_file:
        yaml.dump(data=content, stream=yaml_file)

    config = TestConfig.from_yaml(file_path=full_path)

    for key, value in content.items():
        assert getattr(config, key) == value


def test_yaml_config_from_yaml_errors(tmp_path: Path) -> None:
    """Verifies the error-handling behavior of the YamlConfig class from_yaml() method."""

    @dataclass
    class TestConfig(YamlConfig):
        pass

    invalid_path = tmp_path / "invalid.txt"

    error_message: str = (
        f"Unable to create the dataclass instance using the data from a .yaml file. The 'file_path' argument "
        f"must end in the '.yaml' or '.yml' extension, but got {invalid_path}."
    )

    with pytest.raises(ValueError, match=error_format(error_message)):
        TestConfig.from_yaml(file_path=invalid_path)


@pytest.mark.parametrize(
    "file_contents, expected_type_name",
    [
        ("", "NoneType"),
        ("- 1\n- 2\n- 3\n", "list"),
        ("42\n", "int"),
    ],
)
def test_yaml_config_from_yaml_non_mapping_errors(tmp_path: Path, file_contents: str, expected_type_name: str) -> None:
    """Verifies that the YamlConfig class from_yaml() method raises an error when the file does not contain a
    top-level mapping.

    Verifies the behavior for an empty file (yields None), a sequence document, and a scalar document.
    """

    @dataclass
    class TestConfig(YamlConfig):
        pass

    yaml_path = tmp_path / "non_mapping.yaml"
    yaml_path.write_text(file_contents)

    error_message: str = (
        f"Unable to create the dataclass instance using the data from the {yaml_path} .yaml file. The file "
        f"must contain a top-level mapping, but got {expected_type_name}."
    )

    with pytest.raises(ValueError, match=error_format(error_message)):
        TestConfig.from_yaml(file_path=yaml_path)


def test_yaml_config_initialization() -> None:
    """Verifies the initialization of the YamlConfig class with different input parameters."""

    @dataclass
    class TestConfig(YamlConfig):
        name: str
        count: int
        values: list | None = None

    config = TestConfig(name="test", count=42, values=[1, 2, 3])
    assert config.name == "test"
    assert config.count == 42
    assert config.values == [1, 2, 3]


def test_yaml_config_subclassing() -> None:
    """Verifies the subclassing of the YamlConfig class to provide additional fields."""

    @dataclass
    class ExtendedConfig(YamlConfig):
        extra_param: str
        another_param: dict

    config = ExtendedConfig(extra_param="extra", another_param={"key": "value"})
    assert isinstance(config, YamlConfig)
    assert config.extra_param == "extra"
    assert config.another_param == {"key": "value"}

    # Tests that the subclass still has the 'to_yaml' and 'from_yaml' methods.
    assert hasattr(config, "to_yaml")
    assert hasattr(ExtendedConfig, "from_yaml")


def test_path_round_trip(tmp_path: Path) -> None:
    """Verifies that Path, Path | None, and list[Path] fields round-trip through YAML correctly."""

    @dataclass
    class PathConfig(YamlConfig):
        single_path: Path = Path("/usr/bin")
        nullable_path: Path | None = None
        path_list: list[Path] = field(default_factory=list)

    config = PathConfig(
        single_path=Path("/home/user/data"),
        nullable_path=Path("/opt/output"),
        path_list=[Path("/a/b"), Path("/c/d")],
    )
    yaml_path = tmp_path / "paths.yaml"
    config.to_yaml(file_path=yaml_path)

    # Verifies that raw YAML contains strings, not Path objects.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["single_path"] == "/home/user/data"
    assert raw["nullable_path"] == "/opt/output"
    assert raw["path_list"] == ["/a/b", "/c/d"]

    # Verifies round-trip back to Python produces correct types.
    loaded = PathConfig.from_yaml(file_path=yaml_path)
    assert loaded.single_path == Path("/home/user/data")
    assert isinstance(loaded.single_path, Path)
    assert loaded.nullable_path == Path("/opt/output")
    assert isinstance(loaded.nullable_path, Path)
    assert loaded.path_list == [Path("/a/b"), Path("/c/d")]
    assert all(isinstance(path, Path) for path in loaded.path_list)


def test_path_none_round_trip(tmp_path: Path) -> None:
    """Verifies that a Path | None field with None value round-trips correctly."""

    @dataclass
    class NullablePathConfig(YamlConfig):
        maybe_path: Path | None = None

    config = NullablePathConfig(maybe_path=None)
    yaml_path = tmp_path / "null_path.yaml"
    config.to_yaml(file_path=yaml_path)

    loaded = NullablePathConfig.from_yaml(file_path=yaml_path)
    assert loaded.maybe_path is None


def test_str_enum_round_trip(tmp_path: Path) -> None:
    """Verifies that StrEnum fields and StrEnum | None fields round-trip through YAML correctly."""

    @dataclass
    class EnumConfig(YamlConfig):
        color: Color = Color.RED
        nullable_color: Color | None = None

    config = EnumConfig(color=Color.GREEN, nullable_color=Color.BLUE)
    yaml_path = tmp_path / "enum.yaml"
    config.to_yaml(file_path=yaml_path)

    # Verifies that raw YAML contains string values.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["color"] == "green"
    assert raw["nullable_color"] == "blue"

    # Verifies round-trip deserialization.
    loaded = EnumConfig.from_yaml(file_path=yaml_path)
    assert loaded.color is Color.GREEN
    assert isinstance(loaded.color, Color)
    assert loaded.nullable_color is Color.BLUE


def test_int_enum_round_trip(tmp_path: Path) -> None:
    """Verifies that IntEnum fields are serialized as ints and deserialized back to enum members."""

    @dataclass
    class PriorityConfig(YamlConfig):
        level: Priority = Priority.LOW

    config = PriorityConfig(level=Priority.HIGH)
    yaml_path = tmp_path / "priority.yaml"
    config.to_yaml(file_path=yaml_path)

    # Verifies that raw YAML contains an int.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["level"] == 3
    assert isinstance(raw["level"], int)

    # Verifies round-trip deserialization.
    loaded = PriorityConfig.from_yaml(file_path=yaml_path)
    assert loaded.level is Priority.HIGH
    assert isinstance(loaded.level, Priority)


def test_tuple_round_trip(tmp_path: Path) -> None:
    """Verifies that tuple fields (int tuple, Path tuple, empty tuple) round-trip through YAML correctly."""

    @dataclass
    class TupleConfig(YamlConfig):
        int_tuple: tuple[int, ...] = ()
        path_tuple: tuple[Path, ...] = ()
        empty_tuple: tuple = ()

    config = TupleConfig(
        int_tuple=(1, 2, 3),
        path_tuple=(Path("/a"), Path("/b")),
        empty_tuple=(),
    )
    yaml_path = tmp_path / "tuples.yaml"
    config.to_yaml(file_path=yaml_path)

    # Verifies that raw YAML contains lists.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["int_tuple"] == [1, 2, 3]
    assert raw["path_tuple"] == ["/a", "/b"]
    assert raw["empty_tuple"] == []

    # Verifies round-trip deserialization.
    loaded = TupleConfig.from_yaml(file_path=yaml_path)
    assert loaded.int_tuple == (1, 2, 3)
    assert isinstance(loaded.int_tuple, tuple)
    assert loaded.path_tuple == (Path("/a"), Path("/b"))
    assert isinstance(loaded.path_tuple, tuple)
    assert all(isinstance(path, Path) for path in loaded.path_tuple)
    assert loaded.empty_tuple == ()
    assert isinstance(loaded.empty_tuple, tuple)


def test_nested_dataclass_round_trip(tmp_path: Path) -> None:
    """Verifies that nested dataclasses with Path and Enum fields round-trip through YAML correctly."""

    @dataclass
    class InnerConfig:
        path: Path = Path("/default")
        color: Color = Color.RED

    @dataclass
    class OuterConfig(YamlConfig):
        name: str = ""
        inner: InnerConfig = field(default_factory=InnerConfig)

    config = OuterConfig(name="test", inner=InnerConfig(path=Path("/nested/path"), color=Color.BLUE))
    yaml_path = tmp_path / "nested.yaml"
    config.to_yaml(file_path=yaml_path)

    # Verifies raw YAML structure.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["inner"]["path"] == "/nested/path"
    assert raw["inner"]["color"] == "blue"

    # Verifies round-trip deserialization.
    loaded = OuterConfig.from_yaml(file_path=yaml_path)
    assert loaded.inner.path == Path("/nested/path")
    assert isinstance(loaded.inner.path, Path)
    assert loaded.inner.color is Color.BLUE


def test_union_enum_str_round_trip(tmp_path: Path) -> None:
    """Verifies that Enum | str fields correctly round-trip: valid enum values become enum members, non-enum values
    stay as strings.
    """

    @dataclass
    class UnionConfig(YamlConfig):
        method: Color | str = "auto"

    # Tests with a valid enum value.
    config_enum = UnionConfig(method=Color.RED)
    yaml_path = tmp_path / "union_enum.yaml"
    config_enum.to_yaml(file_path=yaml_path)
    loaded_enum = UnionConfig.from_yaml(file_path=yaml_path)
    assert loaded_enum.method is Color.RED

    # Tests with a non-enum string value.
    config_string = UnionConfig(method="custom_method")
    yaml_path_string = tmp_path / "union_str.yaml"
    config_string.to_yaml(file_path=yaml_path_string)
    loaded_string = UnionConfig.from_yaml(file_path=yaml_path_string)
    assert loaded_string.method == "custom_method"
    assert isinstance(loaded_string.method, str)


def test_primitive_first_union_enum_round_trip(tmp_path: Path) -> None:
    """Verifies that str | Enum and int | Enum (primitive-first) annotations deserialize correctly to enum members.

    Dacite normally tries union members left-to-right, so str | Color would match ``str`` before trying the Color
    hook. The union-level hook registered by ``_collect_type_hooks`` ensures the enum constructor fires first,
    making annotation order irrelevant.
    """

    @dataclass
    class PrimFirstConfig(YamlConfig):
        color: str | Color = Color.RED
        level: int | Priority = Priority.LOW

    config = PrimFirstConfig(color=Color.GREEN, level=Priority.HIGH)
    yaml_path = tmp_path / "prim_first.yaml"
    config.to_yaml(file_path=yaml_path)

    loaded = PrimFirstConfig.from_yaml(file_path=yaml_path)
    assert loaded.color is Color.GREEN
    assert isinstance(loaded.color, Color)
    assert loaded.level is Priority.HIGH
    assert isinstance(loaded.level, Priority)

    # Verifies that non-member values fall back to the primitive type.
    @dataclass
    class PrimFirstFallback(YamlConfig):
        color: str | Color = "auto"

    config_fallback = PrimFirstFallback(color="not_a_color")
    yaml_path_fallback = tmp_path / "prim_first_fallback.yaml"
    config_fallback.to_yaml(file_path=yaml_path_fallback)
    loaded_fallback = PrimFirstFallback.from_yaml(file_path=yaml_path_fallback)
    assert loaded_fallback.color == "not_a_color"
    assert isinstance(loaded_fallback.color, str)


def test_frozen_nested_dataclass_round_trip(tmp_path: Path) -> None:
    """Verifies that a nested frozen dataclass with Path and Enum fields round-trips correctly through YAML."""

    @dataclass(frozen=True)
    class FrozenInner:
        path: Path = Path("/data")
        color: Color = Color.RED

    @dataclass
    class OuterConfig(YamlConfig):
        name: str = "default"
        inner: FrozenInner = field(default_factory=FrozenInner)

    config = OuterConfig(name="frozen_test", inner=FrozenInner(path=Path("/frozen/path"), color=Color.GREEN))
    yaml_path = tmp_path / "frozen.yaml"
    config.to_yaml(file_path=yaml_path)

    loaded = OuterConfig.from_yaml(file_path=yaml_path)
    assert loaded.name == "frozen_test"
    assert loaded.inner.path == Path("/frozen/path")
    assert isinstance(loaded.inner.path, Path)
    assert loaded.inner.color is Color.GREEN


def test_serialize_value_primitives() -> None:
    """Verifies that _serialize_value passes through primitive types unchanged."""
    assert _serialize_value(value=None) is None
    assert _serialize_value(value="hello") == "hello"
    assert _serialize_value(value=42) == 42
    assert _serialize_value(value=3.14) == 3.14
    assert _serialize_value(value=True) is True


def test_serialize_value_path_dict_keys() -> None:
    """Verifies that _serialize_value converts Path keys in dicts to strings."""
    result = _serialize_value(value={Path("/a"): 1, Path("/b"): 2})
    assert result == {"/a": 1, "/b": 2}
    assert all(isinstance(key, str) for key in result)


def test_collect_type_hooks_simple() -> None:
    """Verifies that _collect_type_hooks discovers Path and Enum types in a simple dataclass."""

    @dataclass
    class SimpleConfig(YamlConfig):
        path: Path = Path("/data")
        color: Color = Color.RED
        name: str = ""

    hooks = _collect_type_hooks(cls=SimpleConfig)
    assert Path in hooks
    assert Color in hooks
    assert str not in hooks


def test_collect_type_hooks_nested() -> None:
    """Verifies that _collect_type_hooks discovers types in nested dataclasses."""

    @dataclass
    class Inner:
        priority: Priority = Priority.LOW

    @dataclass
    class Outer(YamlConfig):
        path: Path = Path("/data")
        inner: Inner = field(default_factory=Inner)

    hooks = _collect_type_hooks(cls=Outer)
    assert Path in hooks
    assert Priority in hooks


def test_collect_type_hooks_union_enum() -> None:
    """Verifies that _collect_type_hooks registers union-level hooks for str | Enum and int | Enum annotations."""

    @dataclass
    class UnionConfig(YamlConfig):
        color: str | Color = Color.RED
        level: int | Priority = Priority.LOW

    hooks = _collect_type_hooks(cls=UnionConfig)

    # Concrete enum hooks should still be registered.
    assert Color in hooks
    assert Priority in hooks

    # Union-level hooks should be registered for the union types themselves.
    assert (str | Color) in hooks
    assert (int | Priority) in hooks

    # The union hook should convert valid enum values and pass through non-members.
    str_color_hook = hooks[str | Color]
    assert str_color_hook("red") is Color.RED
    assert str_color_hook("not_a_color") == "not_a_color"
    assert str_color_hook(None) is None

    int_priority_hook = hooks[int | Priority]
    assert int_priority_hook(1) is Priority.LOW
    assert int_priority_hook(99) == 99


def test_collect_type_hooks_skips_a_generic_union_member() -> None:
    """Verifies that _collect_type_hooks registers the enum of a union whose other member is a generic alias."""

    @dataclass
    class GenericUnionConfig(YamlConfig):
        # 'list[str]' is a generic alias rather than a class, so the union walk steps over it to reach the enum.
        items: list[str] | None = None
        tint: Color | None = None

    hooks = _collect_type_hooks(cls=GenericUnionConfig)

    assert Color in hooks
    assert (Color | None) in hooks


class _ImportProxy:
    """Stands in for a module-level import proxy used in place of an absent optional dependency.

    Notes:
        Reporting 'type' as its class is what makes isinstance() accept the proxy as a class, while the absent
        '__bases__' tuple is what makes issubclass() reject it. Annotating a field with such a proxy is the
        realistic way a dataclass reaches the type hook walk's issubclass failure handlers.

        The name attributes are what the dataclass machinery reads when it renders the annotation, which every
        object standing in for a class carries.
    """

    def __init__(self) -> None:
        self.__qualname__ = "_ImportProxy"

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return type

    def __call__(self, *arguments: object, **keywords: object) -> None:
        """Accepts the construction call that typing performs when it validates a union member."""


def test_collect_type_hooks_skips_a_union_member_that_is_not_a_real_class() -> None:
    """Verifies that _collect_type_hooks registers the enum of a union whose other member fails the subclass check."""
    proxy = _ImportProxy()

    # The proxy is not a real class, so the union is built through typing rather than through the '|' operator.
    annotation = Union[Color, proxy]  # noqa: UP007 - The '|' operator rejects a member that is not a real class.

    @dataclass
    class ProxyUnionConfig(YamlConfig):
        tint: annotation = Color.RED  # type: ignore[valid-type]

    hooks = _collect_type_hooks(cls=ProxyUnionConfig)

    # The proxy is stepped over rather than aborting the walk, so the real enum member still earns its hooks.
    assert Color in hooks
    assert hooks[annotation]("red") is Color.RED


def test_collect_type_hooks_returns_no_hook_for_an_annotation_that_is_not_a_real_class() -> None:
    """Verifies that _collect_type_hooks answers with no hook for a field annotated with an import proxy."""

    @dataclass
    class ProxyConfig(YamlConfig):
        value: _ImportProxy() = None  # type: ignore[valid-type]

    assert _collect_type_hooks(cls=ProxyConfig) == {}


def test_collect_type_hooks_returns_no_hook_for_an_unresolvable_annotation() -> None:
    """Verifies that _collect_type_hooks answers with no hook for a class whose annotations do not resolve."""

    @dataclass
    class UnresolvableConfig(YamlConfig):
        # The annotation names a type that exists nowhere, so resolving the class hints raises rather than returning.
        value: "NeverDefinedType" = None  # type: ignore[name-defined]  # noqa: F821 - Intentionally unresolvable.

    assert _collect_type_hooks(cls=UnresolvableConfig) == {}


def test_to_yaml_leaves_previous_file_intact_when_the_dump_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a failed dump removes its temporary file and leaves the previously saved document in place."""

    @dataclass
    class TestConfig(YamlConfig):
        value: str = ""

    yaml_path = tmp_path / "atomic.yaml"
    TestConfig(value="original").to_yaml(file_path=yaml_path)
    original_bytes = yaml_path.read_bytes()

    def failing_dump(**_kwargs: Any) -> None:
        message = "simulated dump failure"
        raise RuntimeError(message)

    monkeypatch.setattr(target="yaml.dump", name=failing_dump)

    with pytest.raises(RuntimeError, match="simulated dump failure"):
        TestConfig(value="replacement").to_yaml(file_path=yaml_path)

    # The previous document survives untouched, and the temporary file the failed write created is gone.
    assert yaml_path.read_bytes() == original_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == ["atomic.yaml"]


def test_yaml_round_trip_uses_utf8_regardless_of_locale(tmp_path: Path) -> None:
    """Verifies that non-ASCII content authored as UTF-8 survives a load, rather than decoding through the locale."""

    @dataclass
    class TestConfig(YamlConfig):
        name: str = ""

    yaml_path = tmp_path / "unicode.yaml"
    yaml_path.write_bytes("---\nname: caf\u00e9 na\u00efve\n...\n".encode())

    assert TestConfig.from_yaml(file_path=yaml_path).name == "caf\u00e9 na\u00efve"


def test_mapping_key_round_trip(tmp_path: Path) -> None:
    """Verifies that Path and Enum mapping keys are restored as their annotated types, matching the write side."""

    @dataclass
    class MappingConfig(YamlConfig):
        by_path: dict[Path, int] = field(default_factory=dict)
        by_color: dict[Color, str] = field(default_factory=dict)
        by_string: dict[str, int] = field(default_factory=dict)

    yaml_path = tmp_path / "mappings.yaml"
    MappingConfig(
        by_path={Path("/a/b"): 1},
        by_color={Color.RED: "warm"},
        by_string={"plain": 2},
    ).to_yaml(file_path=yaml_path)

    # The document stores plain strings, since that is what YAML can express.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw == {"by_path": {"/a/b": 1}, "by_color": {"red": "warm"}, "by_string": {"plain": 2}}

    loaded = MappingConfig.from_yaml(file_path=yaml_path)
    assert loaded.by_path == {Path("/a/b"): 1}
    assert all(isinstance(key, Path) for key in loaded.by_path)
    assert loaded.by_color == {Color.RED: "warm"}
    assert all(isinstance(key, Color) for key in loaded.by_color)
    # A key type needing no conversion is left exactly as YAML produced it.
    assert loaded.by_string == {"plain": 2}
    assert all(isinstance(key, str) for key in loaded.by_string)


def test_mapping_key_hook_passes_through_a_non_mapping() -> None:
    """Verifies that the key hook returns a non-mapping value untouched, which a mistyped document can supply."""
    hooks = _collect_type_hooks(cls=_PathKeyedConfig)
    assert hooks[dict[Path, int]]("not a mapping") == "not a mapping"


@dataclass
class _PathKeyedConfig(YamlConfig):
    """Declares a Path-keyed mapping so the key hook can be retrieved for direct testing."""

    by_path: dict[Path, int] = field(default_factory=dict)


def test_excluded_field_round_trip(tmp_path: Path) -> None:
    """Verifies the pattern a subclass uses to keep a location field out of the document it writes."""

    @dataclass
    class LocatedConfig(YamlConfig):
        value: int = 0
        source_path: Path = field(default=Path("/unset"), metadata=YAML_EXCLUDE_METADATA)

        @classmethod
        def restore_excluded_fields(cls, data: dict[Any, Any], file_path: Path) -> dict[Any, Any]:
            """Reattaches the instance to the file it was read from."""
            return {**data, "source_path": file_path}

    yaml_path = tmp_path / "located.yaml"
    LocatedConfig(value=7, source_path=Path("/the/writing/host/path")).to_yaml(file_path=yaml_path)

    # The writing host's path stays out of the document entirely.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw == {"value": 7}

    # The reader supplies the path from where it found the file, rather than from what the writer recorded.
    loaded = LocatedConfig.from_yaml(file_path=yaml_path)
    assert loaded.value == 7
    assert loaded.source_path == yaml_path
