"""A Python library that provides classes and structures for storing, manipulating and sharing data between Python
processes.

See https://github.com/Sun-Lab-NBB/ataraxis-data-structures for more details.
API documentation: https://ataraxis-data-structures-api-docs.netlify.app/
Authors: Ivan Kondratyev (Inkaros) & Edwin Chen
"""

from .data_loggers import DataLogger
from .shared_memory import SharedMemoryArray
from .data_structures import YamlConfig, NestedDictionary

__all__ = ["SharedMemoryArray", "NestedDictionary", "YamlConfig", "DataLogger"]
