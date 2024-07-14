"""This pacakge provides multiple data structures not available from default / popular Python libraries. This includes
structures that are generally available, but lack the necessary configuration / functionality to work for Sun Lab
projects.

Currently, it exposes the following classes:
    - NestedDictionary: A class that wraps a nested Python dictionary and exposes methods to manipulate the values
        and keys of the dictionary using path-like interface.
    - YamlConfig: A customized dataclass equipped with methods to save and load itself from a .yaml file. This class is
        intended to be used as a parent that provides yaml saving and loading functionality to custom configuration
        dataclasses.

See individual package modules for more details on each of the exposed classes.
"""

from .yaml_config import YamlConfig
from .nested_dictionary import NestedDictionary

__all__ = ["NestedDictionary", "YamlConfig"]
