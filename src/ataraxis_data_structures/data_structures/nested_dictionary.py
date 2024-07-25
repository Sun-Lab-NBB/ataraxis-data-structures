"""This module contains the NestedDictionary class that wraps a nested Python dictionary and exposes methods for
manipulating dictionary keys and values through a path-based interface.

The primary advantage of NestedDictionary class is that it simplifies working with python nested dictionaries without
having a-priori knowledge of their structure. In turn, this is helpful when writing pipelines that should work for a
wide range of under-specified dictionary layouts. Additionally, even for cases where dictionary layout is known, many
methods of the class conveniently simplify complex operations like replacing the datatype of all dictionary keys.
"""

import copy
from types import NoneType
from typing import Any, Type, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console, ensure_list


class NestedDictionary:
    """Wraps a nested (hierarchical) python dictionary and provides methods for manipulating its values.

    This class is primarily designed to abstract working with nested dictionaries without a-priori knowledge of
    dictionary layout. For example, when using data logging methods that produce nested .json or .yml files that can be
    parsed into dictionaries, the data can subsequently be wrapped and processed using this class. Alternatively, this
    class is used in AtaraxisData class to provide a compressed map of the data stored as the unified hdf data to
    optimize data access and manipulation.

    Notes:
        All class methods that modify the wrapped dictionary can be used to either modify the dictionary in-place or to
        return a new NestedDictionary class instance that wraps the modified dictionary.

        While this class will work for both nested and shallow (one-level) dictionaries, it would be inefficient to
        leverage the class machinery for non-nested dictionaries. For shallow dictionaries, using pure python
        methods is preferred.

    Attributes:
        _valid_datatypes: Stores supported dictionary key datatypes as a tuple. The class is guaranteed to recognize
            and work with these datatypes. This variable is used during input checks and for error messages related to
            key datatype conversion errors.
        _nested_dictionary: Stores the managed dictionary object. This object should never be accessed directly!
        _path_delimiter: Stores the sequence used to separate path nodes for input and output dictionary variable paths.
            The paths are used for purposes like accessing specific nested values.
        _key_datatypes: A set that stores the unique string names for the datatypes used by the keys in the dictionary.
            The datatype names are extracted from the __name__ property of the keys, so the class should be able to
            recognize more or less any type of keys. That said, support beyond the standard key datatypes listed in
            valid_datatypes is not guaranteed.

    Args:
        seed_dictionary: The 'seed' dictionary object to be used by the class. If not provided, the class will generate
            an empty shallow dictionary and use that as the initial object. This argument allows to re-initialize nested
            dictionaries when they are loaded from .yaml or .json files, by passing loaded dictionaries as seeds.
        path_delimiter: The delimiter used to separate keys in string variable paths. It is generally advised
            to stick to the default delimiter for most use cases. Only use custom delimiter if any of the dictionary or
            sub-dictionary keys reserve default delimiter for other purposes (for example, if the delimiter is part of a
            string key). Note, all methods in the class refer to this variable during runtime, so all inputs to the
            class have to use the class delimiter where necessary to avoid unexpected behavior.

    Raises:
        TypeError: If input arguments are not of the supported type.
    """

    def __init__(self, seed_dictionary: Optional[dict[Any, Any]] = None, path_delimiter: str = ".") -> None:
        # Stores supported key datatypes
        self._valid_datatypes: tuple[str, str, str, str] = ("int", "str", "float", "NoneType")

        # Verifies input variable types
        if not isinstance(seed_dictionary, (dict, NoneType)):
            message: str = (
                f"A dictionary or None 'nested_dict' expected when initializing NestedDictionary class instance, but "
                f"encountered '{type(seed_dictionary).__name__}' instead."
            )
            console.error(message=message, error=TypeError)

        elif not isinstance(path_delimiter, str):
            message = (
                f"A string 'path_delimiter' expected when initializing NestedDictionary class instance, but "
                f"encountered '{type(path_delimiter).__name__}' instead."
            )
            console.error(message=message, error=TypeError)

        # Sets class attributes:
        # Initial dictionary object
        self._nested_dictionary: dict[Any, Any]
        if seed_dictionary is not None:
            self._nested_dictionary = seed_dictionary  # If it is provided, uses seed dictionary
        else:
            self._nested_dictionary = dict()  # Creates an empty dictionary as the initial object

        # String path delimiter
        self._path_delimiter: str = path_delimiter

        # Sets key_datatype variable to a set that stores all key datatypes. This variable is then used by other
        # methods to support the use of string variable paths (where allowed).
        self._key_datatypes: set[str] = self._extract_key_datatypes()

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""

        id_string: str = (
            f"NestedDictionary(key_datatypes={', '.join(self.key_datatypes)}, "
            f"path_delimiter='{self._path_delimiter}', data={self._nested_dictionary})"
        )
        return id_string

    @property
    def key_datatypes(self) -> tuple[str, ...]:
        """Returns unique datatypes used by dictionary keys as a sorted tuple."""
        # Sorts the keys to make reproducible test results possible.
        return tuple(sorted(self._key_datatypes))

    @property
    def path_delimiter(self) -> str:
        """Returns the delimiter used to separate keys in string variable paths."""
        return self._path_delimiter

    def set_path_delimiter(self, new_delimiter: str) -> None:
        """Sets the path_delimiter class attribute to the provided delimiter value.

        This method can be used to replace the string-path delimiter of an initialized NestedDictionary class.

        Args:
            new_delimiter: The new delimiter to be used for separating keys in string variable paths.

        Raises:
            TypeError: If new_delimiter argument is not a string.

        """
        if not isinstance(new_delimiter, str):
            message = (
                f"A string 'new_delimiter' expected when setting the path delimiter, but "
                f"encountered '{type(new_delimiter).__name__}' instead."
            )
            console.error(message=message, error=TypeError)

        self._path_delimiter = new_delimiter

    def _extract_key_datatypes(self) -> set[str]:
        """Extracts datatype names used by keys in the wrapped dictionary and returns them as a set.

        Saves extracted datatypes as a set and keeps only unique datatype names. Primarily, this information is useful
        for determining whether dictionary variables can be safely indexed using string paths. For example, if the
        length of the set is greater than 1, the dictionary uses at least two unique datatypes for keys and, otherwise,
        the dictionary only uses one datatype. The latter case enables the use of string variable paths, whereas the
        former only allows key iterables to be used as variable paths.

        Returns:
            A set of string-names that describe unique datatypes used by the dictionary keys. The names are extracted
            from each key class using its __name__ property.
        """
        # Discovers and extracts the paths to all terminal variables in the dictionary in raw format
        # (unique, preferred).
        path_keys: tuple[str] | tuple[tuple[Any, ...], ...] = self.extract_nested_variable_paths(return_raw=True)

        # Initializes an empty set to store unique key datatypes
        unique_types: set[str] = set()

        # Loops over all key lists
        for keys in path_keys:
            # Updates the set with the name of the types found in the current key tuple (path)
            unique_types.update(type(key).__name__ for key in keys)

        # Returns extracted key datatype names to caller
        return unique_types

    def _convert_key_to_datatype(
        self, key: Any, datatype: Literal["int", "str", "float", "NoneType"]
    ) -> int | str | float | None:
        """Converts the input key to the requested datatype.

        This method is designed to be used by other class methods when working with dictionary paths and should not
        be called directly by the user.

        Args:
            key: The key to convert to the requested datatype. Generally expected to be one of the standard variable
                types (int, str and float).
            datatype: The string-option that specifies the datatype to convert the key into. Available options are:
                "int", "str", "float" and "NoneType".

        Returns:
            The key converted to the requested datatype.

        Raises:
            ValueError: If the requested datatype is not one of the supported datatypes. If the key value cannot be
                converted to the requested datatype.
        """
        # Matches datatype names to their respective classes using a shallow dictionary to improve the code layout below
        datatypes: dict[str, Type[str | float | int]] = {
            "str": str,
            "int": int,
            "float": float,
        }

        # If datatype is in 'datatypes', and it is not a NoneType, indexes the class out of storage and uses it to
        # convert the key to requested datatype
        if datatype != "NoneType" and datatype in datatypes:
            return datatypes[datatype](key)
        # NoneType datatype is returned as None regardless of the key value
        elif datatype == "NoneType":
            return None
        # If datatype is not found in datatype dictionary, triggers ValueError
        else:
            message: str = (
                f"Unexpected datatype '{datatype}' encountered when converting key '{key}' to the requested "
                f"datatype. Select one of the supported datatypes: {self._valid_datatypes}."
            )
            console.error(message=message, error=ValueError)
            # This is just to appease mypy.
            raise ValueError(message)  # pragma: no cover

    def _convert_variable_path_to_keys(
        self, variable_path: str | NDArray[Any] | tuple[Any, ...] | list[Any]
    ) -> tuple[Any, ...]:
        """Converts the input variable_path to the tuple of keys, which is the format preferred by all class methods.

        This is a utility method not intended to be called from outside the class. It verifies the input variable_path
        in addition to handling the necessary type-conversions to transform the input path into a valid tuple of nested
        key-values. For string variable_path inputs, it converts all keys in the string to the datatype used by the
        dictionary. For tuple, list or numpy array inputs, it assumes that the keys inside the iterable are formatted
        correctly, but checks other iterable properties, such as the number of dimensions.

        Notes:
            Numpy arrays are not valid inputs if the dictionary uses more than a single datatype as they cannot
            represent mixed key types.

        Args:
            variable_path: A string, tuple, list or numpy array that provides the sequence of keys pointing to the
                variable of interest inside the wrapped nested dictionary.

        Returns:
            The tuple of keys that point to a specific unique value in the dictionary. For input string paths, the keys
            are converted to the (only) datatype used by the dictionary. For input key iterables, the input is
            converted into a tuple, but does not undergo datatype-conversion for individual keys.

        Raises:
            TypeError: If the input variable_path is not of a correct type.
            ValueError: If the input variable_path is a string that ends with the class delimiter. If the input
                variable_path is a string or numpy array, and the dictionary keys use more than a single datatype.
                If the input numpy array has more than a single dimension. If the dictionary has an undefined
                key_datatypes property (most often an empty set), likely due to the class wrapping an empty dictionary.
        """

        # For string variable paths, converts the input path keys (formatted as string) into the datatype used by
        # the dictionary keys.
        if isinstance(variable_path, str):
            # If the input argument is a string, ensures it does not end with delimiter.
            if variable_path.endswith(self._path_delimiter):
                message: str = (
                    f"A delimiter-ending variable_path string '{variable_path}' encountered when converting "
                    f"variable path to a sequence of keys, which is not allowed. Make sure the variable path ends "
                    f"with a valid key."
                )
                console.error(message=message, error=ValueError)

            # Additionally, ensures that the string path is accompanied by a valid terminal delimiter value, works
            # equally well for None and any unsupported string options
            elif len(self._key_datatypes) != 1:
                message = (
                    f"An unsupported delimited string variable_path '{variable_path}' encountered when converting "
                    f"variable path to a sequence of keys. To support delimited string inputs, the dictionary has to "
                    f"use a single key datatype, but found {len(self.key_datatypes)} ({self.key_datatypes}) "
                    f"instead. Provide a list or tuple of keys with each key using one of the supported datatypes "
                    f"({self._valid_datatypes})."
                )
                console.error(message=message, error=ValueError)

            # Splits the string path into keys using clas delimiter
            string_keys: list[str] = variable_path.split(self._path_delimiter)

            # Pops the only supported key datatype name out of the storage set to be used below.
            # Deepcopy is used to protect the _key_datatypes attribute from being modified.
            target_dtype = copy.deepcopy(self._key_datatypes).pop()

            # This will raise a ValueError if the conversion fails
            # noinspection PyTypeChecker, LongLine
            keys: str | list[Any] | tuple[Any] | NDArray[Any] = [
                self._convert_key_to_datatype(key=key, datatype=target_dtype)  # type: ignore
                for key in string_keys
            ]

        # For supported iterable path inputs, converts the iterable to a tuple. If individual keys are not valid, this
        # is expected to be caught by the method that called this method.
        elif isinstance(variable_path, (list, tuple, np.ndarray)):
            # For numpy arrays, verifies that the input array has a dimension of 1.
            if isinstance(variable_path, np.ndarray):
                if variable_path.ndim > 1:
                    message = (
                        f"Unable to convert the input variable path numpy array to a tuple of datatype-specific "
                        f"keys when converting variable path to a sequence of keys. Expected a one-dimensional array "
                        f"as input, but encountered an array with unsupported shape ({variable_path.shape}) and "
                        f"dimensionality {variable_path.ndim}."
                    )
                    console.error(message=message, error=ValueError)

                # Additionally, numpy arrays do not support mixed types, so ensures they are only used if the dictionary
                # does not contain mixed key datatypes.
                elif len(self._key_datatypes) != 1:
                    message = (
                        f"An unsupported numpy array variable_path '{variable_path}' encountered when converting "
                        f"variable path to a sequence of keys. To support numpy array inputs, the dictionary has to "
                        f"use a single key datatype, but found {len(self.key_datatypes)} ({self.key_datatypes}) "
                        f"instead. Provide a list or tuple of keys with each key using one of the supported datatypes "
                        f"({self._valid_datatypes})."
                    )
                    console.error(message=message, error=ValueError)

            keys = variable_path
        else:
            message = (
                f"A string, tuple, list or one-dimensional numpy array variable_path expected when "
                f"converting variable path to a sequence of keys. Instead, encountered "
                f"'{type(variable_path).__name__}'. Use one fo the supported variable_path formats."
            )
            console.error(message=message, error=TypeError)

        # noinspection PyUnboundLocalVariable
        return tuple(keys)  # Ensures returned value is a tuple for efficiency

    def extract_nested_variable_paths(
        self,
        *,
        return_raw: bool = False,
    ) -> tuple[str] | tuple[tuple[Any, ...], ...]:
        """Crawls the wrapped nested dictionary and extracts the full path from the top of the dictionary to each
        non-dictionary value.

        The extracted paths can be converted to delimiter-delimited strings or returned as a tuple of key tuples.
        The former format is more user-friendly, but may not contain enough information to fully individuate each pat.
        The latter format allows for each path to be truly unique at the cost of being less user-friendly.

        Notes:
            The output format to choose depends on the configuration of the nested dictionary. If the dictionary only
            contains keys of the same datatype, the delimited strings are the preferred path format and otherwise the
            raw tuple is the preferred format. When this method is called from other NestedDictionary methods, the most
            optimal format is selected automatically.

            This method uses recursive self-calls to crawl the dictionary. This can lead to stack overflow for
            very deep nested dictionaries, although this is not a concern for most use cases.

            This method treats empty sub-dictionaries as valid terminal paths and returns them alongside the paths to
            terminal values.

        Args:
            return_raw: Determines whether the method formats the result as the tuple of key tuples or the tuple of
                delimiter-delimited strings. See notes above for more information.

        Returns:
            If return_raw is true, a tuple of tuples, where each sub-tuple stores a sequence of dictionary path keys.
            If return_raw is false, a tuple of delimiter-delimited path strings.
        """

        def _inner_extract(
            input_dict: dict[Any, Any],
            current_path: Optional[list[Any]] = None,
            *,
            make_raw: bool = False,
        ) -> list[tuple[Any, ...]] | list[str]:
            """Performs the recursive path extraction procedure.

            This sub-method is used to hide recursion variables from end-users, so that they cannot accidentally set
            them to non-default values.

            Most of the extract_nested_variable_paths() method logic comes from this inner method.

            Args:
                input_dict: The dictionary to crawl through. During recursive calls, this variable is used to evaluate
                    sub-dictionaries discovered when crawling the original input dictionary, until the method reaches
                    a non-dictionary value.
                current_path: The ordered list of keys, relative to the top level of the evaluated dictionary. This is
                    used to iteratively construct the sequential key path to each non-dictionary variable. Specifically,
                    recursive method calls add newly discovered keys to the end of the already constructed path key
                    list, preserving the nested hierarchy. This variable is reserved for recursive use, do not change
                    its value!
                make_raw: An alias for the parent method return_raw parameter. This is automatically set to match the
                    parent method return_raw parameter.

            Returns:
                A list of key tuples if return_raw (make_raw) is True and a list of strings otherwise.
            """

            # If current_path is None, creates a new list object to hold the keys. Note, this cannot be a set, as keys
            # at different dictionary levels do not have to be unique, relative to each-other. Therefore, a set may
            # encounter and remove one of the valid duplicated keys along the path. This list is used during recursive
            # calls to keep track of paths being built
            if current_path is None:
                current_path = []

            # This is the overall returned list that keeps track of ALL discovered paths
            paths: list[Any] = []

            # Extracts all items from the current dictionary view
            items = input_dict.items()

            # If the evaluated dictionary is empty, adds it as a terminal path. This allows the method to support
            # dictionaries with empty sub-dictionaries as valid terminal paths. Note, this is only used for
            # sub-dictionaries. If the main dictionary is empty, it will be handled as 'no datatypes' case.
            if len(items) == 0 and len(current_path) != 0:
                paths.append(tuple(current_path) if make_raw else self._path_delimiter.join(map(str, current_path)))
            else:
                # Loops over each key and value extracted from the current view (level) of the nested dictionary
                for key, value in items:
                    # Appends the local level key to the path tracker list
                    new_path = current_path + [key]

                    # If the key points to a dictionary, recursively calls the extract method. Passes the current
                    # path tracker and the dictionary view returned by the evaluated key, to the new method call.
                    if isinstance(value, dict):
                        # The recursion keeps winding until it encounters a non-dictionary variable. Once it does, it
                        # causes the stack to unwind until another dictionary is found via the 'for' loop to start
                        # stack winding. As such, the stack will at most use the same number of instances as the number
                        # of nesting levels in the dictionary, which is unlikely to be critically large.
                        # Note, the 'extend' has to be used here over 'append' to iteratively 'stack' node keys as the
                        # method searches for the terminal variable.
                        # noinspection PyUnboundLocalVariable
                        paths.extend(_inner_extract(input_dict=value, make_raw=make_raw, current_path=new_path))
                    else:
                        # If the key references a non-dictionary variable, formats the constructed key sequence as a
                        # tuple or as a delimited string and appends it to the path list, prior to returning it to
                        # caller. The append operation ensures the path is kept as a separate list object within the
                        # final output list.
                        paths.append(tuple(new_path) if make_raw else self._path_delimiter.join(map(str, new_path)))

            return paths

        # Generates a list of variable paths and converts it to tuple before returning it to the caller. Each path is
        # formatted according to the requested output type by the inner method.
        return tuple(_inner_extract(input_dict=self._nested_dictionary, make_raw=return_raw))  # type: ignore

    def read_nested_value(self, variable_path: str | tuple[Any, ...] | list[Any] | NDArray[Any]) -> Any:
        """Reads the requested value from the nested dictionary using the provided variable_path.

        This method allows accessing individual values stored anywhere across the nested dictionary structure. It can
        return both primitive types and dictionaries of any dimensionality. Therefore, it can be used to slice the
        nested dictionary as needed in addition to reading concrete values.

        Args:
            variable_path: The string specifying the retrievable variable path using the class 'path_delimiter' to
                separate successive keys (nesting hierarchy levels). Example: 'outer_sub_dict.inner_sub_dict.var_1'
                (using dot (.) delimiters). Alternatively, a tuple, list or numpy array of keys that make up the full
                terminal variable path. Example: ('outer_sub_dict', 1, 'variable_6'). Regardless of the input format,
                the path has to be relative to the highest level of the nested dictionary.

        Returns:
            The value retrieved from the dictionary using the provided hierarchical variable path. The value can be a
            variable or a section (dictionary).

        Raises:
            KeyError: If any key in the variable_path is not found at the expected nested dictionary level.
                If a non-terminal key in the key sequence returns a non-dictionary value, forcing the retrieval to
                be aborted prior to fully evaluating the entire variable path.
        """
        # Extracts the keys from the input variable path
        keys: tuple[Any, ...] = self._convert_variable_path_to_keys(variable_path=variable_path)

        # Sets the dictionary view to the highest hierarchy (dictionary itself)
        current_dict_view: Any = self._nested_dictionary

        # Loops over each key in the variable path and iteratively crawls the nested dictionary.
        for num, key in enumerate(keys):
            # If current_dict_view is not a dictionary, but there are still keys to retrieve, issues KeyError and
            # notifies the user the retrieval resulted in a non-dictionary item earlier than expected
            if not isinstance(current_dict_view, dict) and num < len(keys):
                message = (
                    f"Unable to fully crawl the path '{variable_path}', when reading nested value from "
                    f"dictionary. The last used key '{keys[num - 1]}' returned '{current_dict_view}' of type "
                    f"'{type(current_dict_view).__name__}' instead of the expected dictionary type."
                )
                console.error(message=message, error=KeyError)

            # Otherwise, if key is inside the currently evaluated sub-dictionary, uses the key to retrieve the next
            # variable (section or value).
            elif key in current_dict_view:
                current_dict_view = current_dict_view[key]

            # If current_dict_view is a dictionary but the evaluated key is not in dictionary, issues KeyError
            # (key not found)
            else:
                # Generates a list of lists with each inner list storing the value and datatype for each key in
                # current dictionary view
                available_keys_and_types: list[list[str]] = [[k, type(k).__name__] for k in current_dict_view.keys()]

                # Uses the list above ot generate the error message and raises KeyError
                message = (
                    f"Key '{key}' of type '{type(key).__name__}' not found when reading nested value from "
                    f"dictionary using path '{variable_path}'. Make sure the requested key is of the correct "
                    f"datatype. Available keys (and their datatypes) at this level: {available_keys_and_types}."
                )
                console.error(message=message, error=KeyError)

        return current_dict_view

    def write_nested_value(
        self,
        variable_path: str | tuple[Any, ...] | list[Any] | NDArray[Any],
        value: Any,
        *,
        modify_class_dictionary: bool = True,
        allow_terminal_overwrite: bool = True,
        allow_intermediate_overwrite: bool = False,
    ) -> Optional["NestedDictionary"]:
        """Writes the input value to the requested level of the nested dictionary using the provided variable_path.

        This method allows modifying individual values stored anywhere across the nested dictionary structure. It can
        be used to target both terminal values and sections (sub-dictionaries). If any of the keys in the variable_path
        are missing from the dictionary, the method will create and insert new empty sub-dictionaries to add the
        missing keys to the dictionary. This way, the method can be used to set up whole new hierarchies of keys.

        Since the dictionary is modified, rather than re-created, all new subsections will be inserted after existing
        subsections, for each respective hierarchy. For example, when adding 'variable_3' subsection to a section that
        contains 'variable_1, variable_2 and variable_4' (in that order), the result will be:
        'variable_1, variable_2, variable_4, variable_3'.

        Args:
            variable_path: The string specifying the hierarchical path to the variable to be modified / written, using
                the class 'path_delimiter' to separate successive keys (nesting hierarchy levels). Example:
                'outer_sub_dict.inner_sub_dict.var_1' (using dot (.) delimiters). Alternatively, a tuple, list or numpy
                array of keys that make up the full terminal variable path. Example:
                ('outer_sub_dict', 1, 'variable_6'). You can use multiple non-existent keys to specify a new
                hierarchy to add to the dictionary, as each missing key will be used to create an empty section
                (sub-dictionary) within the parent dictionary.
            value: The value to be written. The value is written using the terminal key of the sequence.
            modify_class_dictionary: Determines whether the method will replace the class dictionary
                instance with the modified dictionary generated during runtime (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the original class.
            allow_terminal_overwrite: Determines whether the method is allowed to overwrite already existing terminal
                key values (to replace the values associated with the last key in the sequence).
            allow_intermediate_overwrite: Determines whether the method is allowed to overwrite non-dictionary
                intermediate key values (to replace a variable with a section if the variable is encountered when
                indexing one of the intermediate keys).

        Returns:
            If modify_class_dictionary flag is False, a NestedDictionary instance that wraps the modified dictionary.
            If modify_class_dictionary flag is True, returns None and replaces the class dictionary with the altered
            dictionary.

        Raises:
            KeyError: If overwriting is disabled, but the evaluated terminal key is already in target dictionary.
                If any of the intermediate (non-terminal) keys points to an existing non-dictionary variable and
                overwriting intermediate values is not allowed.
        """
        # Extracts the keys from the input variable path
        keys = self._convert_variable_path_to_keys(
            variable_path=variable_path,
        )

        # Generates a copy of the class dictionary as the method uses modification via reference. This way, the
        # original dictionary is protected from modification while this method runs. Depending on the input
        # arguments, the original dictionary may still be overwritten with the modified dictionary at the end of the
        # method runtime.
        altered_dict: dict[Any, Any] = copy.deepcopy(self._nested_dictionary)
        current_dict_view: dict[Any, Any] = altered_dict

        # Iterates through keys, navigating the dictionary or creating new nodes as needed.
        for num, key in enumerate(keys, start=1):
            # If the evaluated key is the last key in the path sequence, sets the matching value to the value that
            # needs to be written. Due to 'current_dict_view' referencing the input dictionary, this equates to in-place
            # modification.
            if num == len(keys):
                # If the key is not in dictionary, generates a new variable using the key and writes the value to
                # that variable. If the key is already inside the dictionary and overwriting is allowed, overwrites
                # it with new value.
                if key not in current_dict_view or allow_terminal_overwrite:
                    current_dict_view[key] = value

                # The only way to reach this condition is if key is in current dictionary view and overwriting is not
                # allowed, so issues an error.
                else:
                    message = (
                        f"Unable to write the value associated with terminal key '{key}', when writing nested value "
                        f"to dictionary, using path '{variable_path}'. The key already exists at this dictionary level "
                        f"and writing using the key will overwrite the current value of the variable, which is not "
                        f"allowed. To enable overwriting, set 'allow_overwrite' argument to True."
                    )
                    console.error(message=message, error=KeyError)

            # If the evaluated key is not the last key, navigates the dictionary by setting current_dict_view to
            # the sub-dictionary pointed to by the key. If no such sub-dictionary exists, generates and sets an empty
            # sub-dictionary to match the evaluated key.
            else:
                # If key is not in dictionary, generates a new hierarchy (sub-dictionary)
                if key not in current_dict_view:
                    current_dict_view[key] = {}

                # Alternatively, if the key is in dictionary, but it is associated with a variable and not a
                # section, checks if it can be overwritten.
                elif not isinstance(current_dict_view[key], dict):
                    # If allowed, overwrites the variable with an empty hierarchy
                    if allow_intermediate_overwrite:
                        current_dict_view[key] = {}

                    # If not allowed to overwrite, issues an error
                    else:
                        message = (
                            f"Unable to traverse the intermediate key '{key}' when writing nested value to "
                            f"dictionary using variable path '{variable_path}', as it points to a non-dictionary "
                            f"value '{current_dict_view[key]}' and overwriting is not allowed. To enable "
                            f"overwriting, set 'allow_intermediate_overwrite' to True."
                        )
                        console.error(message=message, error=KeyError)

                # Updates current dictionary view to the next level
                current_dict_view = current_dict_view[key]

        # If class dictionary modification is preferred, replaces the wrapped hierarchical dictionary with the altered
        # dictionary
        if modify_class_dictionary:
            self._nested_dictionary = altered_dict

            # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
            # datatypes.
            self._key_datatypes = self._extract_key_datatypes()
            return None
        # Otherwise, constructs a new NestedDictionary instance around the altered dictionary and returns this to
        # caller.
        else:
            return NestedDictionary(seed_dictionary=altered_dict, path_delimiter=self._path_delimiter)

    def delete_nested_value(
        self,
        variable_path: str | tuple[Any, ...] | list[Any] | NDArray[Any],
        *,
        modify_class_dictionary: bool = True,
        delete_empty_sections: bool = True,
        allow_missing: bool = False,
    ) -> Optional["NestedDictionary"]:
        """Deletes the target value from nested dictionary using the provided variable_path.

        This method recursively crawls the nested dictionary hierarchy using the provided variable_path until it
        reaches the terminal key. For that key, deletes the variable or hierarchy (sub-dictionary) referenced by the
        key. If requested, the method can remove hierarchical trees if they were vacated via terminal key deletion,
        potentially optimizing the dictionary structure by removing unused (empty) subdirectories.

        Notes:
            This method uses recursive self-calls to crawl the dictionary. This can lead to stackoverflow for
            very deep nested dictionaries, although this is not a concern for most use cases.

        Args:
            variable_path: The string specifying the hierarchical path to the variable to be deleted, using
                the class 'path_delimiter' to separate successive keys (nesting hierarchy levels). Example:
                'outer_sub_dict.inner_sub_dict.var_1' (using dot (.) delimiters). Alternatively, a tuple, list or
                numpy array of keys that make up the full terminal variable path. Example: ('outer_sub_dict', 1,
                'variable_6').
            modify_class_dictionary: Determines whether the method will replace the class dictionary
                instance with the modified dictionary generated during runtime (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the original class.
            delete_empty_sections: Determines whether dictionary sections made empty by the deletion of underlying
                section / variable keys are also deleted. It is generally recommended to keep this flag set to True to
                optimize memory usage.
            allow_missing: Determine whether missing keys in the variable_path should trigger exceptions. If True,
                missing keys are treated like deleted keys and the method will handle them as if the deletion was
                carried out as expected. If False, the method will notify the user if a particular key is not found in
                the dictionary by raising an appropriate KeyError exception.

        Returns:
            If modify_class_dictionary flag is False, a NestedDictionary instance that wraps the modified dictionary.
            If modify_class_dictionary flag is True, returns None and replaces the class dictionary with the altered
            dictionary.

        Raises:
            KeyError: If any of the target keys are not found at the expected dictionary level, and missing keys are not
                allowed.
        """

        def _inner_delete(
            traversed_dict: dict[Any, Any],
            remaining_keys: list[Any],
            whole_path: tuple[Any, ...] | str,
            *,
            delete_empty_directories: bool,
            missing_ok: bool,
        ) -> None:
            """Performs the recursive deletion procedure.

            This sub-method is used to optimize recursive variable usage and separate recursion variables from
            user-defined input arguments of the main method.

            Notes:
                Relies on Python referencing the same variable throughout all recursions to work, hence why there are
                no explicit return values. All modifications are performed on the same dictionary in-place.

                The primary purpose of the recursion is to support cleanup of emptied dictionary directories, which is
                desirable for memory optimization purposes.

            Args:
                traversed_dict: The dictionary view to process. Each successive method call receives the dictionary
                    sub-slice indexed by one or more already processed intermediate keys from variable_path, which
                    allows progressively crawling the dictionary with each new method call.
                remaining_keys: The remaining keys that have not been processed yet. During each iterative method call
                    the first key in the list is popped out, until only the terminal key is left.
                whole_path: The whole variable path string or tuple. This is only needed for error message purposes and
                    is not explicitly used for processing.
                missing_ok: An alias for the main method 'allow_missing' flag. Determines whether missing keys are
                    treated as if they have been deleted as expected or as exceptions that need to be raised.

            Raises:
                KeyError: If any of the target keys are missing from the evaluated dictionary view, and missing keys are
                    not allowed.
            """

            # If recursion has reached the lowest level, deletes the variable referenced by the terminal key.
            # Note, this step is called only for the lowest level of recursion (terminal key) and for this final step,
            # only this clause is evaluated.
            if len(remaining_keys) == 1:
                final_key: Any = remaining_keys.pop(0)  # Extracts the key from list to variable

                # If the key is found inside the dictionary, removes the variable associated with the key.
                if final_key in traversed_dict:
                    del traversed_dict[final_key]

                # If the final key is not found in the dictionary, handles the situation according to whether
                # missing keys are allowed or not. If missing keys are not allowed, issues KeyError.
                elif not missing_ok:
                    # Generates a list of lists, with each inner list storing the value and datatype for each key in
                    # current dictionary view.
                    available_keys_and_types: list[list[str]] = [[k, type(k).__name__] for k in traversed_dict.keys()]
                    message = (
                        f"Unable to delete the variable matching the final key '{final_key}' of type "
                        f"'{type(final_key).__name__}' from nested dictionary as the key is not found along the "
                        f"provided variable path '{whole_path}'. Make sure the requested key is of the correct "
                        f"datatype. Available keys (and their datatypes) at this level: {available_keys_and_types}."
                    )
                    console.error(message=message, error=KeyError)

                # If the method did not raise an exception, triggers stack unwinding.
                return

            # All further code is executed exclusively for intermediate (non-terminal) recursive instances.
            # Recursion winding up: pops the first path key from the remaining keys list and saves it to a separate
            # variable.
            next_key: Any = remaining_keys.pop(0)

            # If the key is not inside the dictionary, handles the situation according to missing key settings. Either
            # raises a KeyError or triggers stack unwinding.
            if next_key not in traversed_dict:
                # If missing keys are not allowed, raises KeyError
                if not missing_ok:
                    # Generates a list of lists, with each inner list storing the value and datatype for each key in
                    # current dictionary view.
                    available_keys_and_types = [[k, type(k).__name__] for k in traversed_dict.keys()]
                    message = (
                        f"Unable to find the intermediate key '{next_key}' of type '{type(next_key).__name__}' from "
                        f"variable path '{whole_path}' while deleting nested value from dictionary. Make sure the "
                        f"requested key is of the correct datatype. Available keys (and their datatypes) at this "
                        f"level: {available_keys_and_types}."
                    )
                    console.error(message=message, error=KeyError)

                # CRITICAL, if missing keys are allowed, stops stack winding by triggering return and starts stack
                # unwinding even if this did not reach the terminal key. All keys past the key that produced the
                # accepted error are not evaluated and are assumed to be deleted.
                return

            # If next_key is inside the dictionary, carries on with stack winding.
            # Uses remaining_keys that now have one less key due to the popped key. This ensures there is no infinite
            # recursion. Note, the cal to the inner_delete blocks until the terminal key is reached and then essentially
            # works in reverse, where the unblocking travels from the terminal key all the way to the first instance
            # of the method. This is the mechanism that allows cleaning up empty directories introduced by the
            # deletion procedure.
            _inner_delete(
                traversed_dict=traversed_dict[next_key],
                remaining_keys=remaining_keys,
                whole_path=variable_path,  # type: ignore
                missing_ok=allow_missing,
                delete_empty_directories=delete_empty_directories,
            )

            # Stack unwinding: deletes any emptied directories along the path. This cleanup is carried out as the
            # method unwinds from recursion (once the terminal key is reached) for all recursions other than the
            # terminal one, which deletes the last key. If any sub-dictionaries (directories) along the variable path
            # are now (after last/previous key removal) empty, removes it from the dictionary, which may trigger further
            # key removals if this step results in an empty subdirectory.
            if delete_empty_directories and not traversed_dict[next_key]:
                del traversed_dict[next_key]

        # Main method body: applies recursive _inner_delete method to the input dictionary and variable path:

        # Ensures that the evaluated path section uses the tuple format.
        keys: tuple[Any, ...] = self._convert_variable_path_to_keys(
            variable_path=variable_path,
        )

        # Generates a local copy of the dictionary to prevent unwanted modification of the wrapped dictionary.
        processed_dict: dict[Any, Any] = copy.deepcopy(self._nested_dictionary)

        # Initiates recursive processing by calling the first instance of the _inner_delete method. Note, the method
        # modifies the dictionary by reference and has no explicit return statement.
        _inner_delete(
            traversed_dict=processed_dict,
            remaining_keys=list(keys),  # Lists are actually more efficient here as they allow in-place modification.
            whole_path=variable_path,  # type: ignore
            delete_empty_directories=delete_empty_sections,
            missing_ok=allow_missing,
        )

        # If class dictionary modification is preferred, replaces the wrapped dictionary with the modified dictionary.
        if modify_class_dictionary:
            self._nested_dictionary = processed_dict
            # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
            # datatypes.
            self._key_datatypes = self._extract_key_datatypes()

            return None

        # Otherwise, constructs a new NestedDictionary instance around the modified dictionary and returns it to the
        # caller.
        else:
            return NestedDictionary(seed_dictionary=processed_dict, path_delimiter=self._path_delimiter)

    def find_nested_variable_path(
        self,
        target_key: str | int | float | None,
        search_mode: Literal["terminal_only", "intermediate_only", "all"] = "terminal_only",
        *,
        return_raw: bool = False,
    ) -> tuple[tuple[Any, ...] | str, ...] | tuple[Any, ...] | str | None:
        """Extracts the path(s) to the target variable (key) from the wrapped hierarchical dictionary.

        This method is designed to 'find' requested variables and return their paths, so that they can be modified by
        other class methods. This is primarily helpful when no a-priori dictionary layout information is available.

        To do so, the method uses extract_nested_dict_param_paths() method from this class to discover paths to
        each non-dictionary variable and then iterates over all keys in each of the extracted paths until it finds all
        keys that match the 'target_key' argument.

        Notes:
            The method evaluates both the value and the datatype of the input key when searching for matches. If more
            than one match is found for the input target_key, all discovered paths will be returned as a tuple, in the
            order of discovery.

            The output format to choose depends on the configuration of the nested dictionary. If the dictionary only
            contains keys of the same datatype, the delimited strings are the preferred path format and otherwise the
            raw tuple is the preferred format. When this method is called from other NestedDictionary methods, the most
            optimal format is selected automatically.

        Args:
            target_key: A key which points to the value of interest (variable name). Can be a terminal key pointing to
                a variable value or an intermediate key pointing to a sub-dictionary (section). The method will
                account for the input key datatype when searching for the target variable inside the class dictionary.
            search_mode: Specifies the search mode for the method. Currently, supports 3 search modes:
                'terminal_only', 'intermediate_only' and 'all'. 'terminal_only' mode only searches the terminal
                (non-dictionary) keys in each path. 'intermediate_only' mode only searches non-terminal (section /
                dictionary) keys in each path. 'all' searches all keys in each path.
            return_raw: Determines whether the method formats the result as the tuple of key tuples or the tuple of
                delimiter-delimited strings. See notes above for more information.

        Returns:
            If return_raw is true, a tuple of tuples, where each sub-tuple stores a sequence of dictionary path keys.
            If return_raw is false, returns a tuple of delimiter-delimited path strings.
            If only a single matching path was found, returns it as a tuple of keys or a string, depending on the
            value of the return_raw flag.
            If no matching path was found, returns None.

        Raises:
            TypeError: If the input target_key argument is not of the correct type.
            ValueError: If the input search_mode is not one of the supported options.
        """

        # Stores currently supported search modes, which is used for error checking and messaging purposes.
        supported_modes = ("terminal_only", "intermediate_only", "all")

        # Checks that the input key is of the supported type.
        if not isinstance(target_key, (str, int, float, NoneType)):
            message = (
                f"A string, integer, float or NoneType target_key expected when searching for the path to the "
                f"target nested dictionary variable, but encountered '{target_key}' of type "
                f"'{type(target_key).__name__}' instead."
            )
            console.error(message=message, error=TypeError)

        # Ensures that the search mode is one of the supported modes.
        if search_mode not in supported_modes:
            message = (
                f"Unsupported search mode '{search_mode}' encountered when searching for the path to the target "
                f"nested dictionary variable '{target_key}'. Use one of the supported modes: {supported_modes}."
            )
            console.error(message=message, error=ValueError)

        # Extracts all parameter (terminal variables) paths from the dict as a raw tuple.
        var_paths: tuple[tuple[Any, ...], ...] = self.extract_nested_variable_paths(return_raw=True)  # type: ignore

        # Sets up a set and a list to store the data. The set is used for uniqueness checks, and the list is used to
        # preserve the order of discovered keys relative to the order of the class dictionary. This method is
        # chosen for efficiency.
        passed_paths: set[tuple[Any, ...]] = set()
        storage_list: list[tuple[Any, ...]] = []

        # Loops over all discovered variable paths
        for path in var_paths:
            # Adjusts the search procedure based on the requested mode. This conditional assumes that the search_mode
            # validity is verified before this conditional is entered.
            if search_mode == "terminal_only":
                # For terminal_only mode, limits search to the last key of each path.
                keys = path[-1]
                # Since 'terminal_only' mode relies on skipping all keys other than the terminal key, generates a
                # modifier to make the general search procedure below correctly return the whole path despite only
                # evaluating the last key.
                modifier = len(path) - 1
            elif search_mode == "intermediate_only":
                # For 'intermediate_only' mode, removes the terminal keys from the search space.
                keys = path[:-1]
                # Due to how sub-path evaluation is handled, no modifier is needed in this case.
                modifier = 0
            else:
                # For 'all' mode, keeps all keys and also does not use a modifier.
                keys = path
                modifier = 0

            # Carries out the search. This procedure goes through all keys remaining after adjusting the search space
            # and compares each key to the target key. When multiple keys from each path can be evaluated, the procedure
            # works from the highest level to the lowest level of each path. If any key in the sequence matches the
            # target key, the path up to and including the key is saved to the return list.
            for num, key in enumerate(ensure_list(keys), start=1):
                # Note on 'ensure_list' above. For terminal keys, Python automatically optimizes one-element tuples to
                # variables. This breaks the search mechanism here. To address the issue, keys are always cast to list
                # before this loop is executed.
                scanned_path = path[: num + modifier]
                if key == target_key and scanned_path not in passed_paths:
                    passed_paths.add(scanned_path)
                    storage_list.append(scanned_path)  # Preserves order of key discovery

        # If at least one path was discovered, returns a correctly formatted output
        if len(passed_paths) > 0:
            # Raw formatting: paths are returned as tuple(s) of keys
            if return_raw:
                if len(passed_paths) > 1:  # For many paths, returns tuple of tuples
                    return tuple(storage_list)
                else:  # For a single path, returns the path as a tuple of keys
                    return storage_list.pop(0)

            # String formatting: paths are returned as delimited strings
            else:
                # If strings are requested, loops over all discovered path tuples and converts them to
                # class-delimiter-delimited strings
                string_list: list[str] = [self._path_delimiter.join(map(str, path)) for path in storage_list]
                if len(passed_paths) > 1:  # For many paths, returns tuple of strings
                    return tuple(string_list)
                else:  # For a single path, returns the path as a string
                    return string_list.pop(0)

        # Otherwise, returns None to indicate that no matching paths were found.
        else:
            return None

    def convert_all_keys_to_datatype(
        self,
        datatype: Literal["str", "int"],
        *,
        modify_class_dictionary: bool = True,
    ) -> Optional["NestedDictionary"]:
        """Converts all keys inside the class dictionary to use the requested datatype.

        This method is designed to un-mix dictionaries that use multiple key datatypes. Generally, it is preferable for
        dictionaries to use the same datatype (most commonly, string) for all keys. Working with these dictionaries is
        more efficient, and it is possible to use path strings, rather than key tuples, for improved user experience.
        Therefore, successfully running this method on mixed-datatype dictionaries can often lead to better
        user-experience.

        Args:
            datatype: The datatype to convert the dictionary keys to. Currently, only accepts 'int' and 'str'
                string-options as valid arguments, as these are the two most common (and most likely to be successfully
                resolved) datatypes.
            modify_class_dictionary: Determines whether the method will replace the class dictionary
                instance with the modified dictionary generated during runtime (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the original class.

        Returns:
            If modify_class_dictionary flag is False, a NestedDictionary instance that wraps the modified dictionary.
            If modify_class_dictionary flag is True, returns None and replaces the class dictionary with the altered
            dictionary.

        Raises:
            ValueError: If the value for the datatype argument is not a supported datatype string-option.
        """
        valid_datatypes = ("str", "int")  # Stores allowed datatype options, mostly for error messaging

        # Ensures the datatype is one of the valid options.
        if datatype not in valid_datatypes:
            message = (
                f"Unsupported datatype option '{datatype}' encountered when converting the nested dictionary keys "
                f"to use a specific datatype. Select one of the supported options: {valid_datatypes}"
            )
            console.error(message=message, error=ValueError)

        # Retrieves all available dictionary paths as tuples of keys.
        all_paths: tuple[tuple[Any, ...], ...] = self.extract_nested_variable_paths(return_raw=True)  # type: ignore

        # Converts all keys in all paths to the requested datatype.
        try:
            # Uses list comprehension to call key conversion method on each key of each path and casts the resultant
            # list as a tuple.
            converted_paths: tuple[tuple[int | str | float | None, ...], ...] = tuple(
                [tuple(self._convert_key_to_datatype(key=key, datatype=datatype) for key in path) for path in all_paths]
            )
        except Exception as e:
            message = (
                f"Unable to convert dictionary keys to '{datatype}' datatype when converting the nested dictionary "
                f"keys to use a specific datatype. Specifically, encountered the following error: {str(e)}"
            )
            console.error(message=message, error=RuntimeError)

        # Initializes a new nested dictionary class instance using parent class delimiter and an empty seed dictionary.
        converted_dict: NestedDictionary = NestedDictionary(seed_dictionary={}, path_delimiter=self._path_delimiter)

        # Loops over each converted path, retrieves the value associated with the original (pre-conversion) path and
        # writes it to the newly created dictionary using the converted path.
        try:
            # noinspection PyUnboundLocalVariable
            for num, path in enumerate(converted_paths):
                # Retrieves the value using the unconverted path.
                value: Any = self.read_nested_value(
                    variable_path=all_paths[num],
                )

                # Writes the value to the new dictionary using the converted path. Since all overwriting options are
                # disabled, if the conversion resulted in any path duplication or collision, the method will raise an
                # exception
                converted_dict.write_nested_value(
                    variable_path=path,
                    value=value,
                    modify_class_dictionary=True,
                    allow_terminal_overwrite=False,
                    allow_intermediate_overwrite=False,
                )
        except Exception as e:
            message = (
                f"Unable to recreate the dictionary using converted paths when converting the nested dictionary "
                f"keys to use the '{datatype}' datatype. This is most likely because the conversion resulted in having "
                f"at least one pair of duplicated keys at the same hierarchy level. Specific error message: {str(e)}"
            )
            console.error(message=message, error=RuntimeError)

        # If class dictionary modification is preferred, replaces the wrapped class dictionary with the modified
        # dictionary
        if modify_class_dictionary:
            self._nested_dictionary = copy.deepcopy(converted_dict._nested_dictionary)
            # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
            # datatypes
            self._key_datatypes = self._extract_key_datatypes()

            return None
        # Otherwise, returns the newly constructed NestedDictionary instance
        else:
            return converted_dict
