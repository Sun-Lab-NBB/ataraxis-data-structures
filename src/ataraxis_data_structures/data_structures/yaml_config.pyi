from pathlib import Path
from dataclasses import dataclass

@dataclass
class YamlConfig:
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
    @classmethod
    def from_yaml(cls, config_path: Path) -> YamlConfig:
        """Instantiates the class using the data loaded from the provided .yaml (YAML) file.

        This method is designed to re-initialize config classes from the data stored in non-volatile memory. The method
        uses dacite, which adds support for complex nested configuration class structures.

        Args:
            config_path: The path to the .yaml file to read the class data from.

        Returns:
            A new config class instance created using the data read from the .yaml file.

        Raises:
            ValueError: If the provided file path does not point to a .yaml or .yml file.
        """
