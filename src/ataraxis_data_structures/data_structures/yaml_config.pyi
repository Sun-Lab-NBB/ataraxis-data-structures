from pathlib import Path
from dataclasses import dataclass

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
    @classmethod
    def from_yaml(cls, config_path: Path) -> YamlConfig:
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
