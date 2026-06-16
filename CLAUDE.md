# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This builds an accurate model of the project architecture before changes are made, preventing inconsistencies with
the patterns that downstream Ataraxis framework projects depend on.

## Style guide compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load
Ataraxis framework conventions. This applies to ALL file types:

| Task                                | Skill to Invoke    |
|-------------------------------------|--------------------|
| Writing or modifying Python code    | `/python-style`    |
| Writing or modifying README files   | `/readme-style`    |
| Writing git commit messages         | `/commit`          |
| Writing or modifying pyproject.toml | `/pyproject-style` |
| Configuring tox.ini                 | `/tox-config`      |

All contributions must strictly follow these conventions. Key conventions include:
- Google-style docstrings with proper sections
- Full type annotations with explicit array dtypes
- Keyword arguments for function calls
- Third person imperative mood for comments and documentation
- Proper error handling with `console.error()`
- 120 character line limit

## Cross-referenced library verification

Ataraxis framework projects often depend on other `ataraxis-*` libraries. These libraries may be
stored locally in the same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

| Skill                   | Description                                                                    |
|-------------------------|--------------------------------------------------------------------------------|
| `/explore-codebase`     | Perform in-depth codebase exploration at session start                         |
| `/python-style`         | Apply Ataraxis framework Python coding conventions (REQUIRED for code changes) |
| `/readme-style`         | Apply Ataraxis framework README conventions                                    |
| `/commit`               | Draft Ataraxis framework style-compliant git commit messages                   |
| `/pyproject-style`      | Apply Ataraxis framework pyproject.toml conventions                            |
| `/tox-config`           | Apply Ataraxis framework tox.ini conventions                                   |
| `/api-docs`             | Apply Ataraxis framework Sphinx API documentation conventions                  |
| `/audit-facts`          | Fact-check documentation against authoritative source code                     |
| `/audit-style`          | Audit files for style and convention compliance                                |
| `/explore-dependencies` | Build an API snapshot of installed ataraxis dependencies                       |
| `/pr`                   | Draft a style-compliant pull request summary                                   |
| `/project-layout`       | Apply Ataraxis framework project structure conventions                         |
| `/release`              | Draft style-compliant release notes from merged PRs                            |
| `/skill-design`         | Generate and verify skill files and CLAUDE.md instructions                     |

## Project context

This is **ataraxis-data-structures**, a Python library that provides classes and structures for storing, manipulating,
and sharing data between Python processes. The library is part of the Ataraxis ecosystem and serves as a foundational
dependency for other Ataraxis framework projects.

### Key areas

| Directory                                | Purpose                                                  |
|------------------------------------------|----------------------------------------------------------|
| `src/ataraxis_data_structures/`          | Main library source code                                 |
| `src/.../shared_memory/`                 | SharedMemoryArray for process-safe data sharing          |
| `src/.../data_structures/`               | YamlConfig and ProcessingTracker classes                 |
| `src/.../data_loggers/`                  | DataLogger and LogArchiveReader for serialized logging   |
| `src/.../processing/`                    | Checksum, transfer, and interpolation utilities          |
| `tests/`                                 | Test suite (mirrors source structure)                    |
| `docs/`                                  | Sphinx API documentation source                          |

### Architecture

- **SharedMemoryArray**: Wraps NumPy arrays in shared memory buffers for IPC with multiprocessing.Lock for
  process-safety. Requires explicit `connect()`, `disconnect()`, and `destroy()` lifecycle management.
- **YamlConfig**: Base dataclass with YAML serialization via `to_yaml()` and `from_yaml()` class methods. Uses dacite
  for deserialization with type-aware conversions for Path and Enum types.
- **DataLogger**: Runs a logger process with an input Queue for buffering serialized LogPackage data. Uses a watchdog
  thread for monitoring. Saves individual `.npy` files that can be assembled into `.npz` archives.
- **LogArchiveReader**: Reads `.npz` log archives with onset timestamp discovery. Supports batch generation for
  parallel processing workflows via `get_batches()`.
- **ProcessingTracker**: File-based pipeline state tracker using FileLock for multi-process coordination. Manages job
  states (SCHEDULED, RUNNING, SUCCEEDED, FAILED) with search and lifecycle features.
- **Processing Utilities**: Directory checksums (xxHash3-128), parallel directory transfer with integrity verification,
  and time-series interpolation (linear for continuous, last-known-value for discrete data).

### Core components

| Component                    | File                                     | Purpose                                                    |
|------------------------------|------------------------------------------|------------------------------------------------------------|
| SharedMemoryArray            | `shared_memory/shared_memory_array.py`   | Process-safe NumPy array in shared memory                  |
| YamlConfig                   | `data_structures/yaml_config.py`         | Dataclass with YAML serialization                          |
| ProcessingTracker            | `data_structures/processing_tracker.py`  | Pipeline state tracking with file locking                  |
| JobState                     | `data_structures/processing_tracker.py`  | Dataclass for job metadata                                 |
| ProcessingStatus             | `data_structures/processing_tracker.py`  | IntEnum (SCHEDULED, RUNNING, SUCCEEDED, FAILED)            |
| DataLogger                   | `data_loggers/serialized_data_logger.py` | Process-based serialized data logging                      |
| LogPackage                   | `data_loggers/serialized_data_logger.py` | Container for source_id, acquisition_time, serialized_data |
| LogArchiveReader             | `data_loggers/log_archive_reader.py`     | Batch reader for .npz archives                             |
| LogMessage                   | `data_loggers/log_archive_reader.py`     | Container for timestamp_us and payload                     |
| assemble_log_archives        | `data_loggers/serialized_data_logger.py` | Aggregates .npy files into .npz archives                   |
| calculate_directory_checksum | `processing/checksum_tools.py`           | xxHash3-128 directory checksums                            |
| transfer_directory           | `processing/transfer_tools.py`           | Parallel directory copy with verification                  |
| delete_directory             | `processing/transfer_tools.py`           | Parallel directory deletion                                |
| interpolate_data             | `processing/interpolation.py`            | Time-series interpolation                                  |

### Code standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow guidance

Before modifying any component below, review its source file for the current implementation, then follow the
component-specific steps.

**Modifying SharedMemoryArray** (`src/ataraxis_data_structures/shared_memory/shared_memory_array.py`):

1. Understand the multiprocessing.Lock integration for process-safety
2. Maintain the `connect()`/`disconnect()`/`destroy()` lifecycle contract
3. Test with actual multiprocessing scenarios (not just single-process)

**Modifying YamlConfig** (`src/ataraxis_data_structures/data_structures/yaml_config.py`):

1. Understand the dacite integration for nested dataclass conversion
2. Maintain type hook support for Path, Enum, and custom conversions
3. Test with complex nested structures and edge cases

**Modifying DataLogger** (`src/ataraxis_data_structures/data_loggers/serialized_data_logger.py`):

1. Understand the Process/Queue/watchdog thread architecture
2. Maintain the LogPackage format (uint8 source_id, uint64 acquisition_time, uint8 array serialized_data)
3. Test with multiprocessing scenarios and verify proper cleanup

**Modifying ProcessingTracker** (`src/ataraxis_data_structures/data_structures/processing_tracker.py`):

1. Understand the FileLock integration for concurrent access safety
2. Maintain the job state lifecycle (SCHEDULED → RUNNING → SUCCEEDED/FAILED)
3. Test with concurrent access from multiple processes

**Adding processing utilities** (`src/ataraxis_data_structures/processing/`):

1. Review existing utilities for the patterns to follow
2. Follow the same conventions for type hints, docstrings, and error handling
3. Export new functions in `src/ataraxis_data_structures/__init__.py`
4. Add corresponding tests in `tests/processing/`

**Important considerations:**

- This library is a dependency for other Ataraxis framework projects; maintain backwards compatibility
- Use `console.error()` from ataraxis-base-utilities for all error handling
- Use `ataraxis-time` for precision timestamps in logging contexts
- All multiprocessing code uses an explicit spawn context (`get_context("spawn")`) for identical cross-platform behavior
