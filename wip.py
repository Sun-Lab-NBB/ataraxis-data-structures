from ataraxis_data_structures import YamlConfig
from dataclasses import dataclass
from pathlib import Path
import tempfile


# All YamlConfig functionality is accessed via subclassing.
@dataclass
class MyConfig(YamlConfig):
    integer: int = 0
    string: str = 'random'


# Instantiates the test class using custom values that do not match the default initialization values.
config = MyConfig(integer=123, string='hello')

# Saves the instance data to a YAML file in a temporary directory. The saved data can be modified by directly editing
# the saved .yaml file.
tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes.
out_path = Path(tempdir.name).joinpath("my_config.yaml")  # Resolves the path to the output file.
config.to_yaml(file_path=out_path)

# Ensures that the cache file has been created.
assert out_path.exists()

# Creates a new MyConfig instance using the data inside the .yaml file.
loaded_config = MyConfig.from_yaml(file_path=out_path)

# Ensures that the loaded data matches the original MyConfig instance data.
assert loaded_config.integer == config.integer
assert loaded_config.string == config.string
