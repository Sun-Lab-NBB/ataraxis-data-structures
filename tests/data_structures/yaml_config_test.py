"""Contains tests for classes and methods provided by the yaml_config.py module."""

from enum import IntEnum, StrEnum
from typing import Optional
from pathlib import Path
from dataclasses import field, dataclass

import yaml
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import YamlConfig
from ataraxis_data_structures.data_structures.yaml_config import _serialize_value, _collect_type_hooks


@pytest.mark.parametrize(
    "config_path, expected_content",
    [
        (Path("config1.yaml"), {"key1": "value1", "key2": 2, "nested": {}, "list": []}),
        (Path("config2.yml"), {"key1": "", "key2": 0, "nested": {"key": "value"}, "list": [1, 2, 3]}),
        (Path("empty_config.yaml"), {"key1": "", "key2": 0, "nested": {}, "list": []}),  # Test case for empty config
    ],
)
def test_yaml_config_to_yaml(tmp_path, config_path, expected_content):
    """Verifies the functionality of the YamlConfig class to_yaml() method.

    Evaluates the following scenarios:
        0 - Saving a simple key-value pair configuration to a .yaml file.
        1 - Saving a nested configuration with lists to a .yml file.
        2 - Saving an empty configuration to a .yaml file.
    """

    @dataclass
    class TestConfig(YamlConfig):
        """Defines the test dataclass. Each tested config should have the same fields (in the same order)."""

        key1: str = ""
        key2: int = 0
        nested: dict = field(default_factory=dict)
        list: list = field(default_factory=list)

    # Generates and dumps the config as a .yaml file.
    config = TestConfig(**expected_content)
    full_path = tmp_path.joinpath(config_path)
    config.to_yaml(full_path)

    # Verifies that the file was created and contains data.
    assert full_path.exists()
    assert full_path.stat().st_size > 0, f"File {full_path} is empty"

    # Manually reads and verifies the config data.
    with full_path.open("r") as yaml_file:
        loaded_content = yaml.safe_load(yaml_file)
        assert loaded_content == expected_content, f"Expected {expected_content}, but got {loaded_content}"


def test_yaml_config_to_yaml_errors(tmp_path):
    """Verifies the error-handling behavior of the YamlConfig class to_yaml() method."""

    @dataclass
    class TestConfig(YamlConfig):
        pass

    config = TestConfig()
    invalid_path = tmp_path / "invalid.txt"

    error_msg: str = (
        f"Invalid file path provided when attempting to write the dataclass instance to a .yaml file. "
        f"Expected a path ending in the '.yaml' or '.yml' extension as 'file_path' argument, but encountered "
        f"{invalid_path}."
    )

    with pytest.raises(ValueError, match=error_format(error_msg)):
        config.to_yaml(invalid_path)


@pytest.mark.parametrize(
    "config_path, content",
    [
        (Path("config1.yaml"), {"key1": "value1", "key2": 2, "nested": None, "list": None}),
        (Path("config2.yml"), {"nested": {"key": "value"}, "list": [1, 2, 3]}),
    ],
)
def test_yaml_config_from_yaml(tmp_path, config_path, content):
    """Verifies the functionality of the YamlConfig class from_yaml() method.

    Evaluates the following scenarios:
        0 - Loading a simple key-value pair configuration from a .yaml file.
        1 - Loading a nested configuration with lists from a .yml file.
    """

    @dataclass
    class TestConfig(YamlConfig):
        key1: str = ""
        key2: int = 0
        nested: Optional[dict] = None  # noqa: UP045 - field name shadows builtin, X | None fails
        list: Optional[list] = None  # noqa: UP045 - field name shadows builtin, X | None fails

    full_path = tmp_path / config_path
    with full_path.open("w") as yaml_file:
        yaml.dump(content, yaml_file)

    config = TestConfig.from_yaml(full_path)

    for key, value in content.items():
        assert getattr(config, key) == value


def test_yaml_config_from_yaml_errors(tmp_path):
    """Verifies the error-handling behavior of the YamlConfig class from_yaml() method."""

    @dataclass
    class TestConfig(YamlConfig):
        pass

    invalid_path = tmp_path / "invalid.txt"

    error_msg = (
        f"Invalid file path provided when attempting to create the dataclass instance using the data from a "
        f".yaml file. Expected the path ending in the '.yaml' or '.yml' extension as 'file_path' argument, but "
        f"encountered {invalid_path}."
    )

    with pytest.raises(ValueError, match=error_format(error_msg)):
        TestConfig.from_yaml(invalid_path)


def test_yaml_config_initialization():
    """Verifies the initialization of the YamlConfig class with different input parameters."""

    @dataclass
    class TestConfig(YamlConfig):
        param1: str
        param2: int
        param3: list = None

    config = TestConfig(param1="test", param2=42, param3=[1, 2, 3])
    assert config.param1 == "test"
    assert config.param2 == 42
    assert config.param3 == [1, 2, 3]


def test_yaml_config_subclassing():
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


def test_path_round_trip(tmp_path):
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
    config.to_yaml(yaml_path)

    # Verifies that raw YAML contains strings, not Path objects.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["single_path"] == "/home/user/data"
    assert raw["nullable_path"] == "/opt/output"
    assert raw["path_list"] == ["/a/b", "/c/d"]

    # Verifies round-trip back to Python produces correct types.
    loaded = PathConfig.from_yaml(yaml_path)
    assert loaded.single_path == Path("/home/user/data")
    assert isinstance(loaded.single_path, Path)
    assert loaded.nullable_path == Path("/opt/output")
    assert isinstance(loaded.nullable_path, Path)
    assert loaded.path_list == [Path("/a/b"), Path("/c/d")]
    assert all(isinstance(p, Path) for p in loaded.path_list)


def test_path_none_round_trip(tmp_path):
    """Verifies that a Path | None field with None value round-trips correctly."""

    @dataclass
    class NullablePathConfig(YamlConfig):
        maybe_path: Path | None = None

    config = NullablePathConfig(maybe_path=None)
    yaml_path = tmp_path / "null_path.yaml"
    config.to_yaml(yaml_path)

    loaded = NullablePathConfig.from_yaml(yaml_path)
    assert loaded.maybe_path is None


def test_str_enum_round_trip(tmp_path):
    """Verifies that StrEnum fields and StrEnum | None fields round-trip through YAML correctly."""

    @dataclass
    class EnumConfig(YamlConfig):
        color: Color = Color.RED
        nullable_color: Color | None = None

    config = EnumConfig(color=Color.GREEN, nullable_color=Color.BLUE)
    yaml_path = tmp_path / "enum.yaml"
    config.to_yaml(yaml_path)

    # Verifies that raw YAML contains string values.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["color"] == "green"
    assert raw["nullable_color"] == "blue"

    # Verifies round-trip deserialization.
    loaded = EnumConfig.from_yaml(yaml_path)
    assert loaded.color is Color.GREEN
    assert isinstance(loaded.color, Color)
    assert loaded.nullable_color is Color.BLUE


def test_int_enum_round_trip(tmp_path):
    """Verifies that IntEnum fields are serialized as ints and deserialized back to enum members."""

    @dataclass
    class PriorityConfig(YamlConfig):
        level: Priority = Priority.LOW

    config = PriorityConfig(level=Priority.HIGH)
    yaml_path = tmp_path / "priority.yaml"
    config.to_yaml(yaml_path)

    # Verifies that raw YAML contains an int.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["level"] == 3
    assert isinstance(raw["level"], int)

    # Verifies round-trip deserialization.
    loaded = PriorityConfig.from_yaml(yaml_path)
    assert loaded.level is Priority.HIGH
    assert isinstance(loaded.level, Priority)


def test_tuple_round_trip(tmp_path):
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
    config.to_yaml(yaml_path)

    # Verifies that raw YAML contains lists.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["int_tuple"] == [1, 2, 3]
    assert raw["path_tuple"] == ["/a", "/b"]
    assert raw["empty_tuple"] == []

    # Verifies round-trip deserialization.
    loaded = TupleConfig.from_yaml(yaml_path)
    assert loaded.int_tuple == (1, 2, 3)
    assert isinstance(loaded.int_tuple, tuple)
    assert loaded.path_tuple == (Path("/a"), Path("/b"))
    assert isinstance(loaded.path_tuple, tuple)
    assert all(isinstance(p, Path) for p in loaded.path_tuple)
    assert loaded.empty_tuple == ()
    assert isinstance(loaded.empty_tuple, tuple)


def test_nested_dataclass_round_trip(tmp_path):
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
    config.to_yaml(yaml_path)

    # Verifies raw YAML structure.
    with yaml_path.open() as yaml_file:
        raw = yaml.safe_load(yaml_file)
    assert raw["inner"]["path"] == "/nested/path"
    assert raw["inner"]["color"] == "blue"

    # Verifies round-trip deserialization.
    loaded = OuterConfig.from_yaml(yaml_path)
    assert loaded.inner.path == Path("/nested/path")
    assert isinstance(loaded.inner.path, Path)
    assert loaded.inner.color is Color.BLUE


def test_union_enum_str_round_trip(tmp_path):
    """Verifies that Enum | str fields correctly round-trip: valid enum values become enum members, non-enum values
    stay as strings.
    """

    @dataclass
    class UnionConfig(YamlConfig):
        method: Color | str = "auto"

    # Tests with a valid enum value.
    config_enum = UnionConfig(method=Color.RED)
    yaml_path = tmp_path / "union_enum.yaml"
    config_enum.to_yaml(yaml_path)
    loaded_enum = UnionConfig.from_yaml(yaml_path)
    assert loaded_enum.method is Color.RED

    # Tests with a non-enum string value.
    config_string = UnionConfig(method="custom_method")
    yaml_path_string = tmp_path / "union_str.yaml"
    config_string.to_yaml(yaml_path_string)
    loaded_string = UnionConfig.from_yaml(yaml_path_string)
    assert loaded_string.method == "custom_method"
    assert isinstance(loaded_string.method, str)


def test_primitive_first_union_enum_round_trip(tmp_path):
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
    config.to_yaml(yaml_path)

    loaded = PrimFirstConfig.from_yaml(yaml_path)
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
    config_fallback.to_yaml(yaml_path_fallback)
    loaded_fallback = PrimFirstFallback.from_yaml(yaml_path_fallback)
    assert loaded_fallback.color == "not_a_color"
    assert isinstance(loaded_fallback.color, str)


def test_frozen_nested_dataclass_round_trip(tmp_path):
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
    config.to_yaml(yaml_path)

    loaded = OuterConfig.from_yaml(yaml_path)
    assert loaded.name == "frozen_test"
    assert loaded.inner.path == Path("/frozen/path")
    assert isinstance(loaded.inner.path, Path)
    assert loaded.inner.color is Color.GREEN


def test_serialize_value_primitives():
    """Verifies that _serialize_value passes through primitive types unchanged."""
    assert _serialize_value(None) is None
    assert _serialize_value("hello") == "hello"
    assert _serialize_value(42) == 42
    assert _serialize_value(3.14) == 3.14
    assert _serialize_value(value=True) is True


def test_serialize_value_path_dict_keys():
    """Verifies that _serialize_value converts Path keys in dicts to strings."""
    result = _serialize_value({Path("/a"): 1, Path("/b"): 2})
    assert result == {"/a": 1, "/b": 2}
    assert all(isinstance(k, str) for k in result)


def test_collect_type_hooks_simple():
    """Verifies that _collect_type_hooks discovers Path and Enum types in a simple dataclass."""

    @dataclass
    class SimpleConfig(YamlConfig):
        path: Path = Path("/data")
        color: Color = Color.RED
        name: str = ""

    hooks = _collect_type_hooks(SimpleConfig)
    assert Path in hooks
    assert Color in hooks
    assert str not in hooks


def test_collect_type_hooks_nested():
    """Verifies that _collect_type_hooks discovers types in nested dataclasses."""

    @dataclass
    class Inner:
        priority: Priority = Priority.LOW

    @dataclass
    class Outer(YamlConfig):
        path: Path = Path("/data")
        inner: Inner = field(default_factory=Inner)

    hooks = _collect_type_hooks(Outer)
    assert Path in hooks
    assert Priority in hooks


def test_collect_type_hooks_union_enum():
    """Verifies that _collect_type_hooks registers union-level hooks for str | Enum and int | Enum annotations."""

    @dataclass
    class UnionConfig(YamlConfig):
        color: str | Color = Color.RED
        level: int | Priority = Priority.LOW

    hooks = _collect_type_hooks(UnionConfig)

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
