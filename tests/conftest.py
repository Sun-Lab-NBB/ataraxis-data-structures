"""Configures the multiprocessing start method for the entire test session.

The library forces the spawn start method for its own multiprocessing primitives via explicit contexts. This module
mirrors that choice at the global level so that child processes the tests create directly also use spawn on every
platform, ensuring that Linux test runs exercise the same code paths as Windows and macOS.
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
