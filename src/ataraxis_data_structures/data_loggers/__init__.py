"""This package provides the DataLogger class that sets up logger instances running on isolated cores and exposes a
shared Queue object for buffering and piping data from any other Process to the logger cores. Currently, the logger is
only intended to save serialized byte arrays used by other Ataraxis projects (notably: ataraxis-video-system and
ataraxis-transport-layer).

See serialized_data_logger.py for more details on the class and its methods.
"""

from .serialized_data_logger import DataLogger, LogPackage

__all__ = ["DataLogger", "LogPackage"]
