"""This pacakge provides multiple data structures not available from default / popular Python libraries. This includes
structures that are generally available, but lack the necessary configurations to work for Sun Lab projects.

Currently, it exposes the following classes:
    - NestedDictionary: A class that wraps a nested Python dictionary and exposes methods to manipulate the values
        in the dictionary using path-interface.
    - YamlConfig: A dataclass equipped with methods to save and load itself from a .yaml file.

See individual package modules for more details on each of the exposed classes.
"""

from .nested_dictionary import NestedDictionary

__all__ = ["NestedDictionary"]
