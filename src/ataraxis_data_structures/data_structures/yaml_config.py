"""Provides the YamlConfig class, which extends the standard Python 'dataclass' class with methods to cache and retrieve
its data from a .yml (YAML) file.
"""

from typing import Any, Self
from pathlib import Path
from dataclasses import asdict, dataclass

import yaml
from dacite import Config, from_dict
from ataraxis_base_utilities import console, ensure_directory_exists


@dataclass
class YamlConfig:
    """Extends the standard Python dataclass with methods to save and load its data from a .yaml (YAML) file.

    Notes:
        This class is designed to be subclassed by custom dataclasses so that they inherit the YAML saving and loading
        functionality.
    """

    def to_yaml(self, file_path: Path) -> None:
        """Saves the instance's data as the specified .yaml (YAML) file.

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

        # Writes the data to the .yaml file.
        with file_path.open("w") as yaml_file:
            yaml.dump(data=asdict(self), stream=yaml_file, **yaml_formatting)  # type: ignore[call-overload]

    @classmethod
    def from_yaml(cls, file_path: Path) -> Self:
        """Instantiates the class using the data loaded from the provided .yaml (YAML) file.

        Notes:
            This method does not carry out type-checking for the loaded data and may instantiate the class using
            unsupported datatypes, if the input data is not stored in the format expected by the instantiated dataclass.

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

        # Disables built-in dacite type-checking. Primarily, this feature is used to support saving and loading the data
        # as built-in Python types (e.g., str), while the destination dataclass expects it to be one of the non-standard
        # types (e.g., Path).
        class_config = Config(check_types=False)

        # Loads the data from the .yaml file.
        with file_path.open() as yml_file:
            data = yaml.safe_load(yml_file)

        # Converts the imported data to a Python dictionary.
        data_dictionary: dict[Any, Any] = dict(data)

        # Uses dacite to instantiate the class using the imported dictionary.
        # noinspection PyTypeChecker
        return from_dict(data_class=cls, data=data_dictionary, config=class_config)
