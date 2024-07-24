"""This module contains the YamlConfig class, which is an extension of the standard Python dataclass that comes with
methods to save and load itself to / from a .yml (YAML) file.

Primarily, this class is designed to be used for storing configuration data used by other runtimes in non-volatile
memory in a human-readable format. However, it can also be adapted for intermediate-term data storage, if needed.
"""

from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass

import yaml
from dacite import Config, from_dict
from ataraxis_base_utilities import console


@dataclass
class YamlConfig:
    """A Python dataclass bundled with methods to save and load itself from a .yml (YAML) file.

    This class extends the base functionality of Python dataclasses by bundling them with the ability to serialize the
    data into non-volatile memory as .yml files. Primarily, this is used to store configuration information for
    various runtimes, but this can also be adapted as a method of storing data.

    Notes:
        The class is intentionally kept as minimal as possible and does not include built-in data verification.
        You need to implement your own data verification methods if you need that functionality. NestedDictionary
        class from this library may be of help, as it was explicitly designed to simplify working with complex
        dictionary structures, such as those obtained by casting a deeply nested dataclass as a dictionary.
    """

    def to_yaml(self, config_path: Path) -> None:
        """Converts the class instance to a dictionary and saves it as a .yml (YAML) file at the provided path.

        This method is designed to dump the class data into an editable .yaml file. This allows storing the data in
        non-volatile memory and manually editing the data between save / load cycles.

        Args:
            config_path: The path to the .yaml file to write. If the file does not exist, it will be created, alongside
                any missing directory nodes. If it exists, it will be overwritten (re-created). The path has to end
                with a '.yaml' or '.yml' extension suffix.

        Raises:
            ValueError: If the output path does not point to a file with a '.yaml' or '.yml' extension.
        """

        # Defines YAML formatting options. The purpose of these settings is to make yaml blocks more readable when
        # being edited offline.
        yaml_formatting = {
            "default_style": "",  # Use single or double quotes for scalars as needed
            "default_flow_style": False,  # Use block style for mappings
            "indent": 10,  # Number of spaces for indentation
            "width": 200,  # Maximum line width before wrapping
            "explicit_start": True,  # Mark the beginning of the document with ___
            "explicit_end": True,  # Mark the end of the document with ___
            "sort_keys": False,  # Preserves the order of the keys as written by creators
        }

        # Ensures that the output file path points to a .yaml (or .yml) file
        if not config_path.suffix == ".yaml" and not config_path.suffix == ".yml":
            message: str = (
                f"Invalid file path provided when attempting to write the YamlConfig class instance to a yaml file. "
                f"Expected a path ending in the '.yaml' or '.yml' extension, but encountered {config_path}. Provide a "
                f"path that uses the correct extension."
            )
            console.error(message=message, error=ValueError)

        # Ensures that the output directory exists. Co-opts a method used by Console class to ensure log file directory
        # exists.
        # noinspection PyProtectedMember
        console._ensure_directory_exists(config_path)

        # Writes the data to a .yaml file using custom formatting defined at the top of this method.
        with open(config_path, "w") as yaml_file:
            yaml.dump(data=asdict(self), stream=yaml_file, **yaml_formatting)  # type: ignore

    @classmethod
    def from_yaml(cls, config_path: Path) -> "YamlConfig":
        """Instantiates the class using the data loaded from the provided .yaml (YAML) file.

        This method is designed to re-initialize config classes from the data stored in non-volatile memory.
        The method uses dacite, which adds support for complex nested configuration class structures.

        Notes:
            Due to this class aiming to be fairly minimalistic, this method disables built-in dacite type-checking
            before instantiating the class. Therefore, you may need to add explicit type-checking logic for the
            resultant class instance to verify it was instantiated correctly.

        Args:
            config_path: The path to the .yaml file to read the class data from.

        Returns:
            A new config class instance created using the data read from the .yaml file.

        Raises:
            ValueError: If the provided file path does not point to a .yaml or .yml file.
        """

        # Ensures that config_path points to a .yaml / .yml file.
        if not config_path.suffix == ".yaml" and not config_path.suffix == ".yml":
            message: str = (
                f"Invalid file path provided when attempting to create the YamlConfig class instance from a yaml file. "
                f"Expected a path ending in the '.yaml' or '.yml' extension, but encountered {config_path}. Provide a "
                f"path that uses the correct extension."
            )
            console.error(message=message, error=ValueError)

        # Disables built-in dacite type-checking
        class_config = Config(check_types=False)

        # Opens and reads the .yaml file. Note, safe_load may not work for reading python tuples, so it is advised
        # to avoid using tuple in configuration files.
        with open(config_path, "r") as yml_file:
            data = yaml.safe_load(yml_file)

        # Converts the imported data to a python dictionary.
        config_dict: dict[Any, Any] = dict(data)

        # Uses dacite to instantiate the class using the imported dictionary. This supports complex nested structures
        # and basic data validation.
        class_instance = from_dict(data_class=cls, data=config_dict, config=class_config)

        # Uses the imported dictionary to instantiate a new class instance and returns it to caller.
        return class_instance
