"""Contains tests for classes and methods provided by the yaml_config.py module."""

from typing import Optional
from pathlib import Path
from dataclasses import field, dataclass

import yaml
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import YamlConfig


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
