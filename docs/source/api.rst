.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Shared Memory
=============

.. automodule:: ataraxis_data_structures.shared_memory
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
===============

.. automodule:: ataraxis_data_structures.data_structures
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the module constant explicitly, since the automodule directive above discovers module-level data through
   the source of the module it documents and therefore skips a constant this package re-exports. The directive names
   the defining module rather than the package, because autodoc reads the attribute docstring from that module's
   source and falls back to the docstring of the value's own type when it is pointed at the re-exporting package.
.. autodata:: ataraxis_data_structures.data_structures.yaml_config.YAML_EXCLUDE_METADATA

Data Loggers
============

.. automodule:: ataraxis_data_structures.data_loggers
   :members:
   :undoc-members:
   :show-inheritance:

Processing
==========

.. automodule:: ataraxis_data_structures.processing
   :members:
   :undoc-members:
   :show-inheritance:
