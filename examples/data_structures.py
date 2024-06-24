"""
A library that stores all nested-dictionary manipulation functions. These functions are mostly used by
configuration classes to store and manipulate various runtime parameters, but various pipelines may also use these
functions for other purposes that require nested dictionary manipulations.

Notably, also stores ValueValidator class, which is used to extract and validate various nested dictionary values,
primarily for the purpose of validating loaded configuration classes.

Notes:
    Many functions available through this library are recursive! While a use case where this would lead to
    stackoverflow or other recursion error is unlikely, it is not impossible. Be sure to check individual function
    docstrings and implementation if you run into recursion-related errors.
"""
from __future__ import annotations
import sys
import traceback
from dataclasses import asdict
from types import NoneType
from typing import Any, Optional, Literal
import numpy as np
import yaml as yaml
from src.utilities.console import (
    AtaraxisError,
    augment_exception_message,
)
from src.utilities.utils import ensure_list
import copy


class NestedDictionary:
    """Wraps a nested (hierarchical) python dictionary class instance and provides methods for manipulating its values.

    Specifically, contains the methods to extract the paths to all variables inside the dictionary, find the path to
    specific variable(s) and read, write and delete variables (and sections) from the dictionary. All dictionary
    modifications can be used to either modify the wrapped dictionary in-place or to return a new NestedDictionary
    instance that wraps the modified dictionary.

    Note, while this class should work for both nested and shallow (one-level) dictionaries, it would be wasteful to
    leverage the class machinery for non-nested dictionaries.

    Attributes:
        valid_datatypes: Stores all valid dictionary key datatypes as a tuple. These are the datatypes that the class
            is guaranteed to recognize and work with. This variable is used during input checks and for error messages
            related to key datatype conversion errors.
        nested_dict: The nested python dictionary object, which a dictionary that makes use of one or more embedded
            sub-dictionaries. Use this variable to retrieve the wrapped dictionary from the class instance if necessary.
        path_delimiter: The delimiter used to separate keys in string variable paths. It is generally advised
            to stick to the default delimiter for most use cases and only use custom delimiter if any of the keys
            reserve default delimiter for other purposes (for example, if the delimiter is part of a string key).
            Note, all functions in the class refer to this variable during runtime, so all inputs to the class have to
            use the class delimiter where necessary to avoid unexpected behavior. Defaults to '.' (dot).
        key_datatypes: A set that stores the unique string names for the datatypes used by the keys in the dictionary.
            The datatypes names are extracted from the __name__ property of the keys, so the function should be able to
            recognize more or less any type of keys. That said, support beyond the standard key datatypes listed in
            valid_datatypes is not guaranteed.

    Args:
        nested_dict: The nested dictionary to be manipulated by the class. See 'nested_dict' class attribute for
            more details.
        path_delimiter: The string to use as the delimiter in dictionary variable path strings. See 'path_delimiter'
            class attribute for more details.

    Raises:
        TypeError: If the input nested_dict is not a python dictionary or if the input path_delimiter is not a
            string.
        Exception: If extract_key_datatypes() function called as part of the initialization process runs into an
            error and raises an exception.

    Methods:
        extract_key_datatypes: Extracts all unique datatypes used by the dictionary keys and returns them as a set.
        verify_variable_path_input: Verifies input variable_path variable used by many major functions and raises an
            appropriate error if the input is invalid.
        convert_key_to_datatype: Converts the input key to the specified datatype, if the datatype is supported.
        convert_variable_path_to_keys: Converts the input delimited string variable_path to a tuple of keys (the
            preferred format).
        extract_nested_variable_paths: Extracts the nested paths to all non-dictionary variables relative to the top
            level of the dictionary and returns them either as a tuple of delimited strings or a tuple of key tuples.
        read_nested_value: Reads the value from the class dictionary using input variable_path and returns it to caller.
        write_nested_value: Writes the input value to the class dictionary using input variable_path.
        delete_nested_value: Deletes the value from the class dictionary using input variable_path.
        find_nested_variable_path: Finds the path(s) ending with the input key and returns them to user as a tuple of
            delimited strings or a tuple of key tuples.
        convert_all_keys_to_datatype: Converts all keys inside the class dictionary to the specified datatype.
    """

    def __init__(self, nested_dict: dict, path_delimiter: str = ".") -> None:
        # Stores supported key datatypes, mostly used for error messaging purposes
        self.valid_datatypes = ("int", "str", "float", "bool", "NoneType")

        # Verifies input variable types
        if not isinstance(nested_dict, dict):
            custom_error_message = (
                f"A dictionary nested_dict expected when initializing NestedDictionary class instance, but encountered "
                f"'{type(nested_dict).__name__}' instead."
            )
            raise TypeError(custom_error_message)
        elif not isinstance(path_delimiter, str):
            custom_error_message = (
                f"A string path_delimiter expected when initializing NestedDictionary class instance, but encountered "
                f"'{type(path_delimiter).__name__}' instead."
            )
            raise TypeError(custom_error_message)

        # Sets class attributes
        # Dictionary, all operations are preformed on this dictionary object
        self.nested_dict = nested_dict

        # Delimiter to be used in string path strings. Note, all functions in the class will expect the same delimiter
        # to be used by input path strings
        self.path_delimiter = path_delimiter

        # Sets key_datatype variable to a set that stores all key datatypes. This variable is then used by other
        # functions to support the use of string variable paths (where allowed).
        try:
            self.key_datatypes = self.extract_key_datatypes()
        except Exception as e:
            custom_error_message = (
                f"Unable to extract dictionary key datatypes when initializing NestedDictionary class instance."
            )
            augment_exception_message(e=e, additional_message=custom_error_message)
            raise

    def __repr__(self) -> str:
        """Gives the class a nicely formatted representation for print and other functions designed to make use of
        repr calls.

        Returns:
            A nicely formatted string that includes the values of the key attributes of the class to represent the class
            object.
        """
        id_string = (
            f"NestedDictionary(key_datatypes={self.key_datatypes}, path_delimiter='{self.path_delimiter}', "
            f"data={self.nested_dict})"
        )
        return id_string

    def extract_key_datatypes(self, multiprocessing: bool = False) -> set | AtaraxisError:
        """Extracts the datatype names used by keys in dictionary and returns them as a set.

        Note, saves extracted datatypes in a set, so only unique datatypes are kept. If the length of the set is greater
        than 1, the dictionary uses at least two unique datatypes for keys and, otherwise, the dictionary only uses
        one datatype. The latter case enables the use of string variable paths, whereas the former only allows key lists
        to be used as variable paths (see dictionary manipulation functions for details).

        Args:
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            A set of string-names that describe unique datatypes used by the dictionary keys. The names are extracted
            from each datatype class __name__ property.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            Exception: If an unexpected error is encountered or to escalate an exception thrown by one of the internal
                sub-functions.

        """
        handled = False
        # It is very unlikely this function will ever fail, but wrapping it in exception block anyway
        try:
            # Discovers and extracts the paths to all terminal variables in the dictionary in raw (truly unique,
            # preferred format)
            path_keys = self.extract_nested_variable_paths(return_raw=True, multiprocessing=False)

            # Initializes an empty set to store unique key datatypes
            unique_types = set()

            # Loops over all key lists
            for keys in path_keys:
                # Updates the set with the types found in the current key tuple (path)
                unique_types.update(type(key).__name__ for key in keys)

            # Returns extracted key datatypes to caller
            return unique_types
        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = f"Unexpected error when extracting nested dictionary key datatypes."
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def convert_key_to_datatype(
            self, key: Any, datatype: Literal["int", "str", "float", "bool", "NoneType"]
    ) -> int | str | float | bool | None:
        """Converts the input key to the requested datatype.

        Note, this function is expected to only be used as a support function for major methods of the class and,
        therefore, does not contain multiprocessing error handling machinery. It is expected that the caller function
        handles wrapping the errors raised by this function for multiprocessing transmission.

        Args:
            key: The key to convert to the requested datatype. Generally expected to be one of the standard variable
                types (int, str, float and bool).
            datatype: The string-option that specifies the datatype to convert the key into. Available options are:
                "int", "str", "float", "bool" and "NoneType".

        Returns:
            The key converted to the requested datatype.

        Raises:
            ValueError: If the requested datatype is not one of the supported datatypes.
                If the key value cannot be converted to the requested datatype.
        """
        handled = False
        # Matches datatype names to their respective classes using a shallow dictionary for better code layout below
        datatypes = {"str": str, "int": int, "float": float, "bool": bool, "NoneType": None}
        try:
            # If datatype is in datatypes, and it is not a NoneType, indexes the class out of storage and uses it to
            # convert the key to requested datatype
            if datatype in datatypes and datatype != "NoneType":
                return datatypes[datatype](key)
            # NoneType datatype is returned as None regardless of the key value
            elif datatype == "NoneType":
                return None
            # If datatype is not found in datatype dictionary, triggers ValueError
            else:
                handled = False
                custom_error_message = (
                    f"Unexpected datatype '{datatype}' encountered when converting key '{key}' to the requested "
                    f"datatype. Select one of the supported datatypes: {self.valid_datatypes}."
                )
                raise ValueError(custom_error_message)
        except ValueError as e:
            # Intercepts ValueErrors raised by type conversion exceptions and modifies exception messages to contain
            # information about this function. Skips modifying the ValueError raised due to unsupported datatype values,
            # as these errors are already formatted to contain function information
            if not handled:
                # Modifies the error message if the evaluated error is due to conversion failure.
                custom_error_message = (
                    f"Unable to convert key '{key}' of type '{type(key).__name__}' to the requested datatype "
                    f"'{datatype}'. Make sure the value of the key is compatible with the requested datatype."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise

    def convert_variable_path_to_keys(
            self, variable_path: str | np.ndarray | tuple | list, operation_description: str
    ) -> tuple:
        """Converts the input variable_path to the tuple of keys, which is the format preferred by all class functions.

        Verifies the input variable_path in addition to handling the necessary keys and variable type conversions.

        For input string variable_path, converts all keys inside the string to the datatype used by the dictionary. For
        tuple, list or numpy array types, assumes that the keys inside the iterable are formatted correctly, but checks
        other iterable properties, such as the number of dimensions, where appropriate. Note, numpy arrays are not valid
        inputs if the dictionary uses more than a single datatype as they cannot handle mixed key types.

        Additionally, the function is co-purposed to verify the dictionary key datatypes on every call. This is to
        check for unexpected underlying class property modifications that may affect the work of major class functions.
        Since this function is called by all functions where such modifications would cause issues, it makes sense to
        use it to handle potential problematic use cases.

        Note, this function is a utility function designed to reduce the repeated boilerplate code usage in major
        functions. It does not contain multiprocessing error handling support and instead relies on the caller functions
        to wrap exception messages raised by the function where necessary.

        Args:
            variable_path: A string, tuple, list or numpy array that provides the sequence of keys pointing to the
                variable of interest inside the class dictionary object.
            operation_description: The brief description of the purpose of the function that called this function. This
                is the string that goes after the 'when' descriptor in the output string message, which communicates
                what the algorithm was doing when it encountered the raised error. Only used if the algorithm runs into
                an error.

        Returns:
            The tuple of keys that point to a specific unique value in the dictionary. For input string paths, the keys
            are converted to the (only) datatype used by the dictionary keys. For input key Iterables, the input is
            converted into a tuple, but does NOT undergo any key datatype modification.

        Raises:
            TypeError: If the input variable path is not of a correct type
            ValueError: If the input variable_path is a string that ends with the class delimiter.
                If the input variable_path is a string or numpy array and the dictionary keys use more than a single
                datatype.
                If any of the key could not be converted to the correct datatype (for keys supplied via string
                path only).
            RuntimeError: If the input numpy array has more than a single dimension.
                If the dictionary has an undefined key_datatypes property (most often an empty set), likely due to the
                class wrapping an empty dictionary.
            Exception: If an unexpected error is encountered.
        """
        handled = False
        try:
            # Reruns key datatype parsing in case something changed since the last time the function was called
            # (eg: the user has manually reassigned the class dictionary)
            local_key_dtypes = self.extract_key_datatypes(multiprocessing=False)

            # If there is a mismatch between the obtained set and the class property set, sets the class property to
            # new set
            if local_key_dtypes != self.key_datatypes:
                self.key_datatypes = local_key_dtypes

            # For string variable paths, converts the input path keys (formatted as string) into the datatype used by
            # the dictionary keys.
            if isinstance(variable_path, str):
                # If the input argument is a string, ensures it does not end with delimiter
                if variable_path.endswith(self.path_delimiter):
                    custom_error_message = (
                        f"A delimiter-ending variable_path string '{variable_path}' encountered when "
                        f"{operation_description}, which is not allowed. Make sure the variable path ends with a "
                        f"valid key."
                    )
                    raise ValueError(custom_error_message)
                # Additionally, ensures that a string path is accompanied by a valid terminal delimiter value, works
                # equally well for None and any unsupported string options
                elif len(self.key_datatypes) > 1:
                    custom_error_message = (
                        f"A string variable_path '{variable_path}' encountered when {operation_description}, but the "
                        f"dictionary contains mixed key datatypes and does not support string variable "
                        f"path format. Provide a tuple, list or numpy array of keys with each key using one of the "
                        f"supported datatypes ({self.valid_datatypes})."
                    )
                    raise ValueError(custom_error_message)

                # Splits the string path into keys using clas delimiter
                string_keys = variable_path.split(self.path_delimiter)

                # Only runs with the rest of the conversion if there is only a single datatype used by the dictionary
                # keys and raises an error otherwise
                if len(local_key_dtypes) != 0:
                    target_dtype = local_key_dtypes.pop()
                else:
                    custom_error_message = (
                        f"Unable to convert the input variable path string to a tuple of datatype-specific keys when "
                        f"{operation_description}, as the dictionary 'key_datatypes' property is undefined (empty set)."
                    )
                    handled = True
                    raise RuntimeError(custom_error_message)

                # Catches datatype conversion errors
                try:
                    keys = [self.convert_key_to_datatype(key=key, datatype=target_dtype) for key in string_keys]
                except ValueError as e:
                    custom_error_message = (
                        f"Unable to assign the datatypes to keys extracted from variable_path string '{variable_path}' "
                        f"when {operation_description}. Make sure the input path string contains valid "
                        f"keys that can be converted to the datatype '{target_dtype}' used by dictionary keys."
                    )
                    augment_exception_message(e=e, additional_message=custom_error_message)
                    handled = True
                    raise

            # For supported iterable path inputs, simply references the iterable and (see below) converts it to tuple.
            # If keys are not valid, this should be caught by the dictionary crawling function that called this function
            elif isinstance(variable_path, (list, tuple, np.ndarray)):
                # Does some additional processing for numpy arrays
                if isinstance(variable_path, np.ndarray):
                    # Numpy arrays can have too many dimensions, so checks that input array has a dimension of 1
                    if variable_path.ndim > 1:
                        custom_error_message = (
                            f"Unable to convert the input variable path numpy array to a tuple of datatype-specific "
                            f"keys when {operation_description}, as it has too many dimensions {variable_path.ndim}. "
                            f"Only one-dimensional numpy arrays are considered valid inputs."
                        )
                        handled = True
                        raise RuntimeError(custom_error_message)

                    # Additionally, numpy arrays do not support mixed types, so treats them similar to path strings
                    elif len(local_key_dtypes) != 1:
                        custom_error_message = (
                            f"A numpy array variable_path '{variable_path}' encountered when {operation_description}, "
                            f"but the dictionary contains mixed key datatypes and does not support numpy array "
                            f"variable path format. Provide a tuple or list of keys with each key using one of the "
                            f"supported datatypes ({self.valid_datatypes})."
                        )
                        handled = True
                        raise ValueError(custom_error_message)

                keys = variable_path
            else:
                custom_error_message = (
                    f"A string, tuple, list or one-dimensional numpy array variable_path expected when "
                    f"{operation_description}, but encountered '{type(variable_path).__name__}' instead."
                )
                raise TypeError(custom_error_message)

            return tuple(keys)  # Ensures returned value is a tuple for efficiency

        except Exception as e:
            if not handled:
                # Modifies the error message if the evaluated error is due to conversion failure.
                custom_error_message = (
                    f"Unable to convert input variable_path '{variable_path}' to keys, when {operation_description}."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise

    def extract_nested_variable_paths(
            self,
            return_raw: bool = False,
            multiprocessing: bool = False,
    ) -> tuple[str] | tuple[tuple[Any]] | AtaraxisError:
        """Crawls the nested dictionary and extracts the full path from the top level to each non-dictionary value.

        The extracted paths can be converted to delimiter-delimited strings or returned as 'raw' tuple of key tuples.
        The former format is more user-friendly, but may not contain enough information to fully individuate each path,
        while the latter format allows for each path to be truly unique at the cost of being less user-friendly.
        The format to chose depends on the configuration of the nested dictionary. If the dictionary only contains keys
        of the same datatype, the delimited strings are the preferred path format and otherwise the raw tuple is the
        preferred format. When this function is called from other NestedClass functions, the most optimal format is
        selected automatically.

        Notes:
             This function utilizes recursive self-calls to crawl the dictionary. This can lead to stackoverflow for
             very deep nested dictionaries, although this is not a concern for most use cases.

        Args:
            return_raw: Use this flag to determine whether the function returns the raw tuple of key tuples, or the
                delimiter-delimited string. The 'raw' return mode allows to preserve the original datatype of the
                extracted keys, which is useful for many applications, whereas delimiter-delimited strings are more
                user-friendly, but only when all keys in the dictionary are of the same datatype. Defaults to False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            A tuple of delimiter-delimited path strings, each pointing to a particular parameter variable in the class
            nested dictionary.
            A tuple of raw key tuples, each tuple jointly pointing to a particular parameter variable in the class
            nested dictionary, if return_raw flag is set to True.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            Exception: If an unexpected error occurs when extracting parameter-paths from the nested dictionary.
        """

        def _inner_extract(input_dict: dict, make_raw: bool, current_path: Optional[list] = None) -> list:
            """Inner function that performs the recursive path extraction procedure.

            This function is used to hide recursion variables from the user, so that they cannot accidentally set them
            to non-default values. This may be security overkill to be revised in future versions.

            With recent optimizations this is more-or-less the entirety of the function logic.

            Args:
                input_dict: The dictionary to crawl through. During recursive calls, this variable is used to evaluate
                    sub-dictionaries discovered when crawling the original input dictionary, until, eventually, it
                    reaches a non-dictionary value.
                make_raw: An alias for the parent function return_raw parameter. A bit redundant, but avoids implicit
                    referencing. Set it to the value of the parent function's 'return_raw' for the version of this
                    function called by the parent function.
                current_path: The ordered list of keys, relative to the top level of the evaluated dictionary. This is
                    used to iteratively construct the sequential key path to each non-dictionary variable as recursive
                    function calls add newly discovered keys to the end of the already constructed path key list to
                    iteratively build the path. This variable is reserved for recursive use, do not change its value.
                    Defaults to None.

            Returns:
                A list of key tuples if return_raw (make_raw) is True and a list of clas-delimiter-delimited strings
                otherwise.
            """
            # If path is None, creates a new list object to hold the keys. Note, this cannot be a set as keys at
            # different dictionary levels do not have to be unique relative ot each-other and, therefore, a set may
            # encounter and remove one of the valid duplicated keys along the path. This list is used during recursive
            # calls to keep track of paths being built
            if current_path is None:
                current_path = []

            paths = []  # This is the overall returned list that keeps track of ALL discovered paths

            # Loops over each key and value extracted from the current view (level) of the nested dictionary
            for key, value in input_dict.items():
                # Appends the local level key to the path tracker list
                new_path = current_path + [key]

                # If the key points to a dictionary, recursively calls the extract function and passes the current
                # path tracker, alongside the dictionary view returned by evaluated key, to the new function call, so
                # that it can crawl and evaluate the discovered sub-dictionary for path keys and variables
                if isinstance(value, dict):
                    # The recursion keeps winding until it encounters a non-dictionary variable. Once it does, it
                    # causes the stack to un-wind back up until another dictionary is found via the for loop to start
                    # stack winding. As such, the stack will at most employ the same number of function as the number
                    # of nesting levels in the dictionary, which is unlikely yto be critically large.
                    # Note, the 'extend' operation ensures only the lowest (non-dictionary) path is preserved as a
                    # list (generated via the .append() below))
                    paths.extend(_inner_extract(input_dict=value, make_raw=make_raw, current_path=new_path))
                else:
                    # If the key references a non-dictionary variable, formats the constructed key sequence as a tuple
                    # or as a delimited string and appends it to the path list, prior to returning it to caller.
                    # The append operation ensures the path is kept as a separate list object within the final output
                    # list
                    paths.append(tuple(new_path) if make_raw else self.path_delimiter.join(map(str, new_path)))
            return paths

        # The outer block that wraps the inner function and optionally translates paths from lists of keys to delimited
        # strings
        handled = False
        try:
            # Generates a list of variable paths and converts it to tuple before returning it to the user. Each path is
            # formatted according to the requested output type by the inner function
            return tuple(_inner_extract(self.nested_dict, make_raw=return_raw))

        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = f"Unexpected error when extracting variable paths from nested dictionary."
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def read_nested_value(
            self,
            variable_path: str | tuple | list | np.ndarray,
            replace_none_equivalents: bool = False,
            multiprocessing: bool = False,
    ) -> str | int | float | list | tuple | dict | bool | None | AtaraxisError | ValueConverter | Any:
        """Reads a value from the nested dictionary using the sequence of nested keys (variable path).

        The variable path can either be a class-delimiter-delimited string of keys or a tuple, list or numpy array of
        keys. The latter input type will always be accepted as valid as it allows to precisely identify the single
        correct retrieval path. The former input will be rejected if the dictionary keys use more than a single unique
        datatype, as path strings do not contain enough information to select the specific required path when keys can
        have different datatypes.

        The function contains a mechanism that replaces upper and lower case string-values 'null' and 'none' with python
        None, but only if replace_none_equivalents flag is True.

        Args:
            variable_path: The string specifying the retrievable variable path using the class 'path_delimiter' to
                separate successive keys (nesting hierarchy levels). Example: 'outer_sub_dict.inner_sub_dict.var_1'
                (using dot (.) delimiters). Alternatively, a tuple, list or numpy array of keys that make up the full
                terminal variable path. Example: ('outer_sub_dict', 1, 'variable_6').
            replace_none_equivalents: Use this flag to determine whether the function should replace common string
                equivalents of pythonic None value (null, none strings, both lower and upper case) with pythonic None.
                Defaults to False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            The value retrieved from the dictionary using provided hierarchical variable path. The value can be a
            variable or a section (dictionary).
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            KeyError: If any key in the variable_path is not found at the expected nested dictionary level.
                If non-terminal key in the key sequence returns a non-dictionary value, forcing the retrieval to
                be aborted prior to fully evaluating the entire key sequence.
            Exception: If an unexpected exception occurs.
        """

        # Stores all None aliases that are expected to be encountered (for conversion purposes)
        none_values = {"None", "Null", "null", "none"}

        handled = False
        try:
            # Extracts the keys from the input variable path
            handled = True
            keys = self.convert_variable_path_to_keys(
                variable_path=variable_path, operation_description="reading nested value from dictionary"
            )
            handled = False

            # Sets the dictionary view to the highest hierarchy (dictionary itself)
            current_dict_view = self.nested_dict

            # Loops over each key in the variable path and iteratively crawls the nested dictionary
            for num, key in enumerate(keys):
                # If current_dict_view is not a dictionary, but there are still keys to retrieve, issues KeyError and
                # notifies the user the retrieval resulted in a non-dictionary item earlier than expected
                if not isinstance(current_dict_view, dict) and num < (len(keys)):
                    custom_error_message = (
                        f"Unable to fully crawl the path '{variable_path}' when reading nested value from "
                        f"dictionary, as last used key '{keys[num - 1]}' returned '{current_dict_view}' of type "
                        f"'{type(current_dict_view).__name__}' instead of the expected dictionary."
                    )
                    handled = True
                    raise KeyError(custom_error_message)

                # Otherwise, if key is inside the currently evaluated sub-dictionary, uses the key to retrieve the next
                # variable (section or value).
                elif key in current_dict_view:
                    current_dict_view = current_dict_view[key]

                # If current_dict_view is a dictionary but the evaluated key is not in dictionary, issues KeyError
                # (key not found)
                else:
                    # Generates a list of lists with each inner list storing the value and datatype for each key in
                    # current dictionary view
                    available_keys_and_types = [[k, type(k).__name__] for k in current_dict_view.keys()]

                    # Provides a custom error message
                    custom_error_message = (
                        f"Key '{key}' of type '{type(key).__name__}' not found when reading nested value from "
                        f"dictionary using path '{variable_path}'. Make sure the requested key is of the correct "
                        f"datatype. Available keys (and their datatypes) at this level: {available_keys_and_types}."
                    )
                    handled = True
                    raise KeyError(custom_error_message)

            # Replaces non-equivalents with a pythonic None value. This is an important step for many other functions
            # that expect a pythonic None as a valid input.
            # If the extracted variable is a string, is in none_values and replace_none_equivalents flag is True,
            # returns None
            if isinstance(current_dict_view, str) and current_dict_view in none_values and replace_none_equivalents:
                return None
            else:
                # Otherwise, returns the extracted value
                return current_dict_view

        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = (
                    f"Unexpected error when reading nested value from dictionary using path '{variable_path}'."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def write_nested_value(
            self,
            variable_path: str | tuple | list | np.ndarray,
            value: Any,
            modify_class_dictionary: bool = True,
            allow_terminal_overwrite: bool = True,
            allow_intermediate_overwrite: bool = False,
            multiprocessing: bool = False,
    ) -> NestedDictionary | None | AtaraxisError:
        """Writes the input value to the nested dictionary using a sequence of nested keys (variable_path).

        The variable path can either be a class-delimiter-delimited string of keys or a tuple, list or numpy array of
        keys. The latter input type will always be accepted as valid as it allows to precisely identify the single
        correct retrieval path. The former input will be rejected if the dictionary keys use more than a single unique
        datatype, as path strings do not contain enough information to select the specific required path when keys can
        have different datatypes.

        If any of the keys in the variable_path are missing from the dictionary, the function will create and insert
        new empty sub-dictionaries to add the missing keys to the dictionary. This way, the function can be used to
        setup whole hierarchies of keys. Since the dictionary is modified, rather than re-created, all new directories
        are inserted after already existing directories (for each respective hierarchy level).

        Args:
            variable_path: The string specifying the hierarchical path to the variable to be modified / written, using
                the class 'path_delimiter' to separate successive keys (nesting hierarchy levels). Example:
                'outer_sub_dict.inner_sub_dict.var_1' (using dot (.) delimiters). Alternatively, a tuple, list or numpy
                array of keys that make up the full terminal variable path. Example:
                ('outer_sub_dict', 1, 'variable_6'). Note, you can use multiple non-existent keys to specify a new
                hierarchy to add to the dictionary, as each missing key will be used to create an empty section
                (sub-dictionary) within the parent dictionary.
            value: The value to be written. The value is written using the terminal key of the sequence.
            modify_class_dictionary: Use this flag to determine whether the function will replace the class dictionary
                instance with the modified dictionary generated by the function (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the parent class of the function. Defaults to
                False.
            allow_terminal_overwrite: Use this flag to determine whether the algorithm is allowed to overwrite already
                existing terminal key values (to replace the values associated with the last key in the sequence) or
                not. Defaults to True.
            allow_intermediate_overwrite: Use this flag to determine whether the algorithm is allowed to overwrite
                non-dictionary intermediate key values (to replace a variable with a section, if the variable is
                encountered when indexing one of the intermediate keys). Defaults to False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            A NestedDictionary instance that wraps the modified dictionary that contains the over/written value and any
            additional hierarchy sections / levels that were necessary to fully realize the variable_path leading to
            the written value, if modify_class_dictionary flag is False.
            Does not return anything, but replaces the class dictionary with the altered dictionary, if
            modify_class_dictionary flag is True.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            RuntimeError: If overwriting is disabled, but the evaluated terminal key is already in target dictionary.
            KeyError: If any of the intermediate (non-terminal) keys points to an existing non-dictionary variable and
                overwriting intermediate values is not allowed.
            Exception: If an unexpected error occurs or to escalate errors generated by inner functions.
        """
        handled = False
        try:
            # Extracts the keys from the input variable path
            handled = True
            keys = self.convert_variable_path_to_keys(
                variable_path=variable_path, operation_description="writing nested value to dictionary"
            )
            handled = False

            # Generates a copy of the class dictionary as the algorithm uses modification via reference. This way the
            # original dictionary is protected from modification while this function runs. Depending on the function
            # arguments, the original dictionary may still be overwritten with the modified dictionary at the end of the
            # function
            altered_dict = copy.deepcopy(self.nested_dict)
            current_dict_view = altered_dict

            # Iterates through keys, navigating the dictionary or creating new nodes as needed
            for num, key in enumerate(keys, start=1):
                # If the evaluated key is the last key in sequence, sets the matching value to the value that needs to
                # be written. Due to 'current_dict_view' referencing the input dictionary, this equates to in-place
                # modification
                if num == len(keys):
                    # If the key is not in dictionary, generates a new variable using the key and writes the value to
                    # that variable. If the key is already inside the dictionary and overwriting is allowed, overwrites
                    # it with new value
                    if key not in current_dict_view or allow_terminal_overwrite:
                        current_dict_view[key] = value

                    # The only way to reach this condition is if key is in dictionary and overwriting is not allowed,
                    # so issues an error
                    else:
                        custom_error_message = (
                            f"Unable to write the value associated with terminal key '{key}' when writing nested value "
                            f"to dictionary, using path '{variable_path}'. The key already exists in dictionary "
                            f"and writing using the key will overwrite the current value of the variable, which is not "
                            f"allowed. To enable overwriting, set 'allow_overwrite' argument to True."
                        )
                        handled = True
                        raise RuntimeError(custom_error_message)

                # If the key is not the last key, either navigates the dictionary by setting current_dict_view to the
                # target subdictionary or, if no such subdictionary exists, generates and sets an empty subdictionary to
                # match the evaluated key.
                else:
                    # If key is not in dictionary, generates a new hierarchy (sub-dictionary)
                    if key not in current_dict_view:
                        current_dict_view[key] = {}
                    # Alternatively, if the key is in dictionary, but it is associated with a variable and not a
                    # section, checks if it can be overwritten
                    elif not isinstance(current_dict_view[key], dict):
                        # IF allowed, overwrites the variable with an empty hierarchy
                        if allow_intermediate_overwrite:
                            current_dict_view[key] = {}
                        # If not allowed to overwrite, issues an error
                        else:
                            custom_error_message = (
                                f"Unable to traverse the intermediate key '{key}' when writing nested value to "
                                f"dictionary using variable path '{variable_path}', as it points to a non-dictionary "
                                f"value '{current_dict_view[key]}' and overwriting is not allowed. To enable "
                                f"overwriting, set 'allow_intermediate_overwrite' to True."
                            )
                            handled = True
                            raise KeyError(custom_error_message)

                    # Updates current dictionary view to the next level
                    current_dict_view = current_dict_view[key]

            # If class dictionary modification is preferred, replaces the bundled class dictionary with the altered
            # dictionary
            if modify_class_dictionary:
                self.nested_dict = altered_dict
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self.key_datatypes = self.extract_key_datatypes()
            # Otherwise, constructs a new NestedDictionary instance around the altered dictionary and returns this to
            # caller
            else:
                return NestedDictionary(nested_dict=altered_dict, path_delimiter=self.path_delimiter)
        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = (
                    f"Unexpected error when writing nested value to dictionary using path '{variable_path}'."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def delete_nested_value(
            self,
            variable_path: str | tuple | list | np.ndarray,
            modify_class_dictionary: bool = True,
            delete_empty_sections: bool = True,
            allow_missing: bool = False,
            multiprocessing: bool = False,
    ) -> NestedDictionary | None | AtaraxisError:
        """Deletes the target value from nested dictionary using a sequence of nested keys (variable_path).

        The variable path can either be a class-delimiter-delimited string of keys or a tuple, list or numpy array of
        keys. The latter input type will always be accepted as valid as it allows to precisely identify the single
        correct retrieval path. The former input will be rejected if the dictionary keys use more than a single unique
        datatype, as path strings do not contain enough information to select the specific required path when keys can
        have different datatypes.

        This function recursively crawls the nested dictionary hierarchy using the provided variable path until it
        reaches the terminal key. For that key, deletes the variable or hierarchy (sub-dictionary) referenced by the
        key. The function uses recursion to iteratively call itself for each progressive key in the sequence. Once it
        deletes the terminal key, the function than un-winds the stack generated through iterative self-calls and for
        each un-winding step checks whether the directory referenced by the key processed by the function became empty
        (for example, when the terminal key is deleted, the pre-terminal directory may become empty). If the directory
        was made empty and the delete_empty_sections flag is True, the function deletes the directory (and the matching
        key), which may in turn vacate higher hierarchy directories. This way, the function can remove whole
        hierarchical trees if they were vacated via terminal key deletion, potentially optimizing the dictionary
        structure by removing unused (empty) subdirectories.

        Notes:
            This function utilizes recursive self-calls to crawl the dictionary. This can lead to stackoverflow for
            very deep nested dictionaries, although this is not a concern for most use cases.

        Args:
            variable_path: The string specifying the hierarchical path to the variable to be deleted, using
                the class 'path_delimiter' to separate successive keys (nesting hierarchy levels). Example:
                'outer_sub_dict.inner_sub_dict.var_1' (using dot (.) delimiters). Alternatively, a tuple, list or
                numpy array of keys that make up the full terminal variable path. Example: ('outer_sub_dict', 1,
                'variable_6').
            modify_class_dictionary: Use this flag to determine whether the function will replace the class dictionary
                instance with the modified dictionary generated by the function (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the parent class of the function. Defaults to
                False.
            delete_empty_sections: Use this flag to determine whether dictionary sections made empty by the deletion of
                underlying section / variable keys are deleted during stack unwinding. It is generally recommend to
                keep this flag set to True to optimize memory usage. Defaults to True.
            allow_missing: Use this flag to determine whether missing keys in the variable_path trigger exceptions or
                are handled gracefully. If True, missing keys are treated like deleted keys and the function will handle
                them as if the deletion was carried out as expected. If False, the function will notify the user if a
                particular key is not found in the dictionary by raising an appropriate KeyError exception.
                Defaults to False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            A NestedDictionary class instance that wraps the truncated dictionary where, minimally, the variable
            referenced by the last key in the provided path and, maximally, the entire path branch has been removed
            (depends on whether the operation made all path directories empty or not).
            Does not return anything, but replaces the class dictionary with the altered dictionary, if
            modify_class_dictionary flag is True.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            KeyError: If any of the target keys are not found at the expected dictionary level and missing keys are not
                allowed.
            Exception: If an unexpected error occurs while attempting to delete the specified node path from dictionary.
        """

        def _inner_delete(
                traversed_dict: dict,
                remaining_keys: list,
                whole_path: tuple | str,
                delete_empty_directories: bool,
                missing_ok: bool,
        ) -> None:
            """Inner function that performs the recursive deletion procedure.

            This function is used to optimize recursive variable usage and separate recursion variables from
            user-defined input arguments of the main function.

            Note, the function relies on python referencing the same variable throughout all recursions to work, hence
            why there are no explicit return values. All modifications are performed on the same dictionary in-place.

            The primary purpose of recursion is to support cleanup of emptied dictionary directories, which is desirable
            for memory optimization purposes.

            Args:
                traversed_dict: The dictionary view to work with. Each successive function call receives the dictionary
                    sub-slice indexed by one or more already processed intermediate keys from variable_path, which
                    allows to progressively crawl the dictionary with each new function call.
                remaining_keys: The remaining keys that have not been processed yet. During each iterative function call
                    the first key in the list is popped out, until, only the terminal key is left.
                whole_path: The whole variable path string or tuple. This is only needed for error message purposes and
                    is not explicitly used for processing.
                missing_ok: The toggle that determines whether missing keys are treated as if they have been deleted as
                    expected or as exceptions that need to be raised.

            Raises:
                KeyError: If any of the target keys are missing from the evaluated dictionary view and missing keys are
                    not allowed.
            """
            # If recursion has reached the lowest level, deletes the variable referenced by the terminal key.
            # Note, this step is called only for the lowest level of recursion (terminal key) and for this final step
            # only this clause is evaluated
            if len(remaining_keys) == 1:
                final_key = remaining_keys.pop(0)  # Extracts the key from list to variable

                # If the key is found inside the dictionary, removes the variable associated with the key
                if final_key in traversed_dict:
                    del traversed_dict[final_key]

                # If the final key is not found in the dictionary, handles the situation according to whether
                # missing keys are allowed or not. If missing keys are not allowed, issues KeyError
                elif not missing_ok:
                    # Generates a list of lists with each inner list storing the value and datatype for each key in
                    # current dictionary view
                    available_keys_and_types = [[k, type(k).__name__] for k in traversed_dict.keys()]
                    inner_error_message = (
                        f"Unable to delete the variable matching the final key '{final_key}' of type "
                        f"'{type(final_key).__name__}' from nested dictionary as the key is not found along the "
                        f"provided variable path '{whole_path}'. Make sure the requested key is of the correct "
                        f"datatype. Available keys (and their datatypes) at this level: {available_keys_and_types}."
                    )
                    raise KeyError(inner_error_message)

                # Triggers stack unwinding (if exception was not raised)
                return

            # All further code is executed exclusively for intermediate (non-terminal) recursive instances.
            # Recursion winding up: pops the first path key from the remaining keys list and saves it to a separate
            # variable
            next_key = remaining_keys.pop(0)

            # If the key is not inside the dictionary, handles the situation according to missing key settings
            if next_key not in traversed_dict:
                # If missing keys are not allowed, raises KeyError
                if not missing_ok:
                    # Generates a list of lists with each inner list storing the value and datatype for each key in
                    # current dictionary view
                    available_keys_and_types = [[k, type(k).__name__] for k in traversed_dict.keys()]
                    inner_error_message = (
                        f"Unable to find the intermediate key '{next_key}' of type '{type(next_key).__name__}' from "
                        f"variable path '{whole_path}' while deleting nested value from dictionary. Make sure the "
                        f"requested key is of the correct datatype. Available keys (and their datatypes) at this "
                        f"level: {available_keys_and_types}."
                    )
                    raise KeyError(inner_error_message)

                # CRITICAL, if missing keys are allowed, stops stack winding by triggering return and starts stack
                # unwinding even if this did not reach the terminal key. All keys past the key that produced the
                # accepted error are not evaluated and are assumed to be deleted
                return

            # If next_key is inside the dictionary, carries on with stack winding.
            # Uses remaining_keys that now have one less key due to popped key. This ensures there is no infinite
            # recursions.
            # Note, this call blocks until the terminal key is reached and then essentially works in reverse, where
            # the un-blocking travels from the terminal key all the way to the first instance of the function
            _inner_delete(
                traversed_dict=traversed_dict[next_key],
                remaining_keys=remaining_keys,
                whole_path=variable_path,
                missing_ok=allow_missing,
                delete_empty_directories=delete_empty_directories,
            )

            # Recursion un-winding: deletes any emptied directories along the path.
            # This cleanup is carried out as the function unwinds from recursion (once the terminal key is reached)
            # for all recursions other than the terminal one, which deletes the last key.
            # If any sub-dictionaries (directories) along the variable path are now (after last/previous key removal)
            # empty, removes it from main dict, which may trigger further key removals if this step results in an
            # empty subdirectory.
            # Note, empty directory cleanup is only carried out if the function is instructed to do so
            if delete_empty_directories and not traversed_dict[next_key]:
                del traversed_dict[next_key]

        # Main function body: applies recursive inner function to the in[put dictionary and variable path
        handled = False
        try:
            # Extracts the keys from the input variable path
            handled = True
            keys = self.convert_variable_path_to_keys(
                variable_path=variable_path, operation_description="deleting nested value from dictionary"
            )
            handled = False

            # Generates a local copy of the dictionary
            processed_dict = copy.deepcopy(self.nested_dict)

            # Initiates recursive processing by calling the first instance of the inner function. Note, the function
            # modifies the dictionary by reference, hence no explicit return statement
            _inner_delete(
                traversed_dict=processed_dict,
                remaining_keys=list(keys),  # Lists are actually more efficient here as they allow in-place modification
                whole_path=variable_path,
                missing_ok=allow_missing,
                delete_empty_directories=delete_empty_sections,
            )

            # If class dictionary modification is preferred, replaces the bundled class dictionary with the processed
            # dictionary
            if modify_class_dictionary:
                self.nested_dict = processed_dict
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self.key_datatypes = self.extract_key_datatypes()
            # Otherwise, constructs a new NestedDictionary instance around the processed dictionary and returns this to
            # caller
            else:
                return NestedDictionary(nested_dict=processed_dict, path_delimiter=self.path_delimiter)

        # Handles KeyErrors. KeyErrors are generated by the recursive inner function and propagated all the way to the
        # main function level, where they have to be handled before the general Exception clause is evaluated.
        except KeyError:
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

        # Handles all unexpected errors excluding KeyError
        except Exception as e:
            if not handled:
                custom_error_message = (
                    f"Unexpected error when deleting nested value from dictionary using '{variable_path}' path."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def find_nested_variable_path(
            self,
            target_key: str | int | float | bool | None,
            search_mode: Literal["terminal_only", "intermediate_only", "all"] = "terminal_only",
            return_raw: bool = False,
            multiprocessing: bool = False,
    ) -> tuple[tuple | str] | tuple | str | None | AtaraxisError:
        """Extracts the path(s) to the target variable (key) from the input hierarchical dictionary.

        This function is designed to extract the path to the target variable stored inside the input hierarchical
        dictionary. To do so, it uses extract_nested_dict_param_paths() function from this class to discover paths to
        each non-dictionary variable and then iterates over all keys in each of the extracted paths until it finds all
        keys that match the 'target_key' argument. Note, the function evaluates both the value and the datatype of the
        input key when searching for matches. If more than one match is found for the input target_key, all discovered
        paths will be returned as a tuple, in the order of discovery.

        The function can be configured to either search only the terminal (variable) keys, only the intermediate
        (section) keys or all available keys for each path.

        Note, you can use 'return_raw' flag to specify whether this function should return a list of key lists
        (generally desired, as this method preserves enough information for each path to point to a single value only).
        If the flag is set to 'False', the function will instead return a list of class-delimiter-delimited string
        paths, which are more user-friendly, but DO NOT contain enough information to ensure they point to a single
        variable, unless the dictionary keys all use the same datatype.

        Args:
            target_key: A key which points to the value of interest (variable name). Can be a terminal key pointing to
                a variable value or an intermediate key pointing to a sub-dictionary (section). Note, the function
                expects the input key to have one of the standard datatypes (int, str, float, bool, NoneType) and will
                account for the input key datatype when searching for the target variable inside the class dictionary.
            search_mode: Specifies the search mode for the algorithm. Currently, supports 3 search modes:
                'terminal_only', 'intermediate_only' and 'all'. 'terminal_only' mode only searches the terminal
                (non-dictionary) keys in each path, 'intermediate_only' mode only searches non-terminal (section /
                dictionary) keys in each path and 'all' searches all keys in each path. Defaults to 'terminal_only'.
            return_raw: Use this flag to determine whether the function returns the raw list of key lists, or the
                delimiter-delimited string. The 'raw' return mode allows to preserve the original datatype of the
                extracted keys, which is useful for many applications, whereas delimiter-delimited strings are more
                user-friendly, but only when all keys in the dictionary are of the same datatype. Defaults to False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            The tuple of key tuples, with each inner tuple representing a sequence of keys ending with the input
            target_key, if return_raw flag is True.
            A tuple of class-delimiter-delimited path strings, if return_raw flag is False.
            None, if the target key is not found in the class nested dictionary.
            A (single) tuple of keys or delimited string, if only a single path was discovered for the input target_key.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            TypeError: If the input target_key argument are not of the correct type.
                If the input search_mode argument is not of a correct type
            ValueError: If the input search mode is not one of the supported options.
            Exception: If an unexpected error is encountered during runtime.
        """
        handled = False
        supported_modes = ("terminal_only", "intermediate_only", "all")
        try:
            # Checks that the input key is of the supported type
            if not isinstance(target_key, (str, int, bool, float, NoneType)):
                custom_error_message = (
                    f"A string, integer, boolean, float or NoneType target_key expected when finding the path to the "
                    f"target nested dictionary variable, but encountered '{target_key}' of type "
                    f"'{type(target_key).__name__}' instead."
                )
                handled = True
                raise TypeError(custom_error_message)

            # Checks that the search_mode is of the correct type
            if not isinstance(search_mode, str):
                custom_error_message = (
                    f"A string search_mode expected when finding the path to the target nested dictionary variable, "
                    f"but encountered '{search_mode}' of type '{type(search_mode).__name__}' instead."
                )
                handled = True
                raise TypeError(custom_error_message)

            # Extracts all parameter (terminal variables) paths from the dict as a raw tuple
            var_paths = self.extract_nested_variable_paths(return_raw=True, multiprocessing=False)

            # Sets up a set and a list to store the data. The set is used for uniqueness checks and the list is used to
            # preserve the order of discovered keys relative to the order of the class dictionary. This method is
            # chosen for efficiency
            passed_paths = set()
            storage_list = []

            # Loops over each extracted path key tuple and checks keys against the target key
            for path in var_paths:
                # If the function is configured to only evaluate terminal keys, only checks the last key for each path
                # tuple
                if search_mode == "terminal_only":
                    # Checks whether the last key matches the target key
                    if path[-1] == target_key:
                        # If terminal key matches target key, verifies that the path is not already in the storage
                        # list and tracker set and, if not, adds it to both set and list
                        if path not in passed_paths:
                            passed_paths.add(path)
                            storage_list.append(path)  # Preserves order of key discovery

                # If the function is configured to evaluate all keys, evaluates each key in each path list and keeps
                # ALL unique paths that lead to the key of interest
                elif search_mode == "all":
                    for num, key in enumerate(path):
                        if key == target_key:
                            # If any key inside the path tuple match the target key, extracts the portion of the path
                            # ending with the target key and adds it to storage list.
                            # Note, the process will iterate through the entire key sequence even if an intermediate
                            # match is found prior to reaching the end of the sequence. This way, if the same key is
                            # used as a section key and as a variable key (for a variable inside that section), both
                            # will be returned to caller.
                            path_keys = path[: num + 1]  # Since slicing is uses exclusive end index, uses num+1

                            # Adds unique paths to the storage list. This ensures that the path is not repeatedly
                            # added for multiple paths originating from the same section (when section key matches the
                            # target key).
                            if path_keys not in passed_paths:
                                passed_paths.add(path_keys)
                                storage_list.append(path_keys)  # Preserves order of key discovery
                # If the function is configured to evaluate intermediate (section) keys only, uses a logic similar to
                # above, but indexes the terminal key out of each evaluated tuple to exclude it from search
                elif search_mode == "intermediate_only":
                    for num, key in enumerate(path[:-1]):
                        if key == target_key:
                            path_keys = path[: num + 1]
                            if path_keys not in passed_paths:
                                passed_paths.add(path_keys)
                                storage_list.append(path_keys)  # Preserves order of key discovery

                # If search_mode is not one of the supported options, triggers an error
                else:
                    custom_error_message = (
                        f"Unsupported search mode '{search_mode}' encountered when finding the path to the target "
                        f"nested dictionary variable. Use one of the supported modes: {supported_modes}."
                    )
                    handled = True
                    raise ValueError(custom_error_message)

            # If at least one path was discovered, returns a correctly formatted output tuple
            if len(passed_paths) > 0:
                # Raw formatting: paths are returned as tuples of keys
                if return_raw:
                    passed_paths = [path for path in storage_list]
                    if len(passed_paths) > 1:  # For many paths, returns tuple of tuples
                        return tuple(passed_paths)
                    else:  # For a single path, returns the path as a tuple or string
                        return passed_paths.pop(0)

                # String formatting: paths are returned as delimited strings
                else:
                    # If strings are requested, loops over all discovered path tuples and converts them to
                    # class-delimiter-delimited strings
                    passed_paths = [self.path_delimiter.join(map(str, path)) for path in storage_list]
                    if len(passed_paths) > 1:  # For many paths, returns tuple of tuples
                        return tuple(passed_paths)
                    else:  # For a single path, returns the path as a tuple or string
                        return passed_paths.pop(0)

            # Otherwise, returns None to indicate that no matching paths were found
            else:
                return None

        except Exception as e:
            if not handled:
                custom_error_message = (
                    f"Unexpected error when finding the path to the target nested dictionary variable."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def convert_all_keys_to_datatype(
            self,
            datatype: Literal["str", "int"],
            modify_class_dictionary: bool = True,
            multiprocessing: bool = False,
    ) -> NestedDictionary | None | AtaraxisError:
        """Converts all keys inside the class dictionary to use the input datatype.

        This function is designed to un-mix dictionaries that make use of multiple datatypes for keys. Generally,
        dictionaries that use the same datatype (most commonly, string) for all keys are preferred as working with
        these dictionaries is less computationally complex, and it is possible to use path strings, rather than key
        tuples, for improved user experience.

        Args:
            datatype: The datatype to convert the dictionary keys to. Currently, only accepts 'int' and 'str'
                string-options as valid arguments, as these are the two most common (and most likely to be successfully
                resolved) datatypes.
            modify_class_dictionary: Use this flag to determine whether the function will replace the class dictionary
                instance with the modified dictionary generated by the function (if True) or generate and return a new
                NestedDictionary instance built around the modified dictionary (if False). In the latter case, the new
                class will inherit the 'path_separator' attribute from the parent class of the function. Defaults to
                False.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            A NestedDictionary class instance that wraps the modified dictionary where all keys have been converted to
            the requested datatype.
            Does not return anything, but replaces the class dictionary with the modified dictionary, if
            modify_class_dictionary flag is True.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            TypeError: If the input datatype variable is not a string.
            ValueError: If the value of the datatype variable is not a supported datatype string-option.
            Exception: If an unexpected error is encountered during runtime or to escalate the errors produced by lower
                level functions.
        """
        handled = False
        valid_datatypes = ("str", "int")  # Stores allowed datatype options, mostly for error messaging
        try:
            # Ensures that the requested datatype variable is of a correct type and value
            if not isinstance(datatype, str):
                custom_error_message = (
                    f"A string datatype argument expected when converting the nested dictionary keys to use a "
                    f"specific datatype, but encountered '{datatype}' of type '{type(datatype).__name__}' instead."
                )
                handled = True
                raise TypeError(custom_error_message)
            elif datatype not in valid_datatypes:
                custom_error_message = (
                    f"Unsupported datatype option '{datatype}' encountered when converting the nested dictionary keys "
                    f"to use a specific datatype. Select one of the supported options: {valid_datatypes}"
                )
                handled = True
                raise ValueError(custom_error_message)

            # Retrieves all available dictionary paths as lists of keys
            all_paths = self.extract_nested_variable_paths(return_raw=True, multiprocessing=False)

            # Converts all keys in all paths to the requested datatype
            try:
                # noinspection PyTypeChecker
                converted_paths = (
                    tuple(self.convert_key_to_datatype(key=key, datatype=datatype) for key in path)
                    for path in all_paths
                )
                converted_paths = tuple(converted_paths)  # Converts the outer iterable into a tuple of tuples

            except Exception as e:
                custom_error_message = (
                    f"Unable to convert dictionary keys to '{datatype}' datatype when converting the nested dictionary "
                    f"keys to use a specific datatype."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
                handled = True
                raise

            # Initializes a new nested dictionary class instance using parent class delimiter and an empty dictionary
            converted_dict = NestedDictionary(nested_dict={}, path_delimiter=self.path_delimiter)

            # Loops over each converted path, retrieves the value associated with original (pre-conversion) path and
            # writes it to the newly created dictionary using the converted path
            try:
                for num, path in enumerate(converted_paths):
                    # Retrieves the value using unconverted path. Note, ensures None-equivalents are NOT converted
                    value = self.read_nested_value(
                        variable_path=all_paths[num], replace_none_equivalents=False, multiprocessing=False
                    )

                    # Writes the value to the new dictionary using converted path.
                    # Note, since all overwrite options are disabled, if the conversion resulted in any path duplication
                    # or collision, the function will raise an exception
                    converted_dict.write_nested_value(
                        variable_path=path,
                        value=value,
                        modify_class_dictionary=True,
                        allow_terminal_overwrite=False,
                        allow_intermediate_overwrite=False,
                        multiprocessing=False,
                    )
            except Exception as e:
                custom_error_message = (
                    f"Unable to recreate the dictionary using converted paths when converting the nested dictionary "
                    f"keys to use a specific datatype, most likely because the conversion resulted in having at least "
                    f"one pair of duplicated keys at the same hierarchy level."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
                handled = True
                raise

            # If class dictionary modification is preferred, replaces the bundled class dictionary with the processed
            # dictionary
            if modify_class_dictionary:
                self.nested_dict = copy.deepcopy(converted_dict.nested_dict)
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self.key_datatypes = self.extract_key_datatypes()
            # Otherwise, returns the newly constructed NestedDictionary instance
            else:
                return converted_dict

        except Exception as e:
            if not handled:
                custom_error_message = (
                    f"Unexpected error when converting the nested dictionary keys to use a specific datatype."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error


class ValueConverter:
    """After initial configuration, allows to conditionally validate and convert input values to the desired datatype.

    The primary use for this class is to convert input values to the datatype(s) defined during class configuration.
    During conversion, input values are checked against the validation parameters of the class prior to being converted
    to the requested datatype, which allows to flexibly and precisely define the range of 'accepted' input and output
    values. This allows the class to serve as both a value converter and validator.

    The primary application for this class is to assist configuration classes, which store data on disk between runtimes
    and, typically, convert all data into string format. This class can be used to convert the strings loaded by
    configuration classes back into the intended format. The class itself can be written and loaded from disk, acting
    as a repository of correct validation / conversion parameters that can be stored in non-volatile memory and used to
    restore the rest of the data to the originally intended datatype.

    Additionally, this class can be used by UI elements to validate user inputs in cases where UI libraries do not
    provide a reliable input validation mechanism.

    Note, the class is designed to be as input-datatype agnostic as possible. In most cases, if a precise input value
    datatype is known, it is more efficient (and easier) to implement a simple in-code conversion. This class is best
    suited for cases when the input value type can vary widely during runtime and/or includes many possible options.

    Attributes:
        allow_bool: Use this flag to allow the validated value to be a boolean or a boolean-equivalent string. Only
            allows boolean-equivalents if parse_bool_equivalents flag is True. Note, if enabled, string 'True'/'False'
            are always parsed as boolean values, but string and integer '1'/'0' values will be preferentially converted
            to integers or floats if allow_int / allow_float and parse_number_strings are also enabled
            (due to datatype preference order).
        allow_string: Use this flag to allow the validated value to be a string. By itself, the flag unconditionally
            validates all string inputs, but string_options (see below) attribute can be used to limit the allowed
            strings to a set of options.
        allow_int: Use this flag to allow the validated value to be an integer or an integer-convertible string or
            float. Integer-convertible strings are only allowed if parse_number_strings flag is True. When enabled
            together with allow_float, the algorithm always tries to convert floats into integers where possible.
            The range of allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        allow_float: Use this flag to allow the validated value to be a float or a float-convertible string or integer.
            Float-convertible strings are only allowed if parse_number_strings flag is True. When enabled together
            with allow_int, the algorithm always tries to convert floats into integers where possible. The range of
            allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        allow_none: Use this flag to allow the validated value to be a None (NoteType) or a None-equivalent string.
            Only allows none-equivalents if parse_none_equivalents flag is True.
        allow_iterables: Use this flag to allow the validated value to be a shallow tuple, list or set. Only supports
            shallow iterables as the algorithm calls the verification function onto each value in the iterable,
            which will fail for any value that is not a string, float, integer, boolean or NoneType.
        allow_string_conversion: Use this flag to enable converting non-string inputs to strings. Since all supported
            input values can be converted to strings, this is a dangerous option that has the potential of overriding
            all verification parameters. It is generally advised to not enable this flag for most use cases.
            Defaults to False.
        string_options: Optional. A tuple, list or set of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
            Set to None to disable string option-limiting. Defaults to None.
        string_force_lower: Use this flag to force validated string values to be converted to lower-case.
            Only used if allow_string is True and only applies to strings. Defaults to False.
        number_lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit. Defaults to None.
        number_upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit. Defaults to None.
        parse_bool_equivalents: Use this flag to enable converting supported boolean-equivalent strings to boolean
            datatype and verifying such strings as boolean values.
        parse_none_equivalents: Use this flag to enable converting supported none-equivalent strings to NoneType (None)
            datatype and verifying such strings as None values.
        parse_number_strings: Use this flag to enable converting all integer- and float-equivalent strings to the
            appropriate numeric datatype and verifying such strings as integers and/or floats.
        iterable_output_type: Optional. A string-option that allows to convert input iterable values to a particular
            iterable type prior to returning them. Only used when input values are iterables. Valid options
            are 'set', 'tuple' and 'list'. Alternatively, set to None to force the algorithm to use the same iterable
            type for output value as for the input value. Defaults to None.
        supported_iterables: Internal use only. This dictionary maps supported iterable classes to their string-names.
            The dictionary is used internally to support output iterable type setting and for checks and error messages.
        true_equivalents: Internal use only. This set specifies boolean True string and integer equivalents. If
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        false_equivalents: Internal use only. Same as true_equivalents, but for boolean False equivalents.
        none_equivalents: Internal use only. Same as true_equivalents, but stores None (NoneType) equivalent strings.

    Raises:
        ValueError: If the input string_options argument is not a set, tuple, list or None.
            Also, if the input string_options argument is an empty set, tuple, list.
            If the input iterable_output_type argument is not one of the supported iterable output types.

    Methods:
        convert_value: The master function of the class. Sets-up the validation and conversion procedure for all input
            value types (iterables and non-iterables) and returns the converted value to caller. This is the only method
            that should be called externally, the rest of the clas methods are designed for internal class use only.
        validate_value: The central validation function that calls the rest of the class validation functions to
            determine whether the input value can be parsed as any of the supported (and allowed) datatypes. Also
            contains the logic that select the most preferred datatype to convert the value to if it can represent
            multiple allowed datatypes.
        check_num: Checks whether the input value is a valid integer or float (or number-parsable string) and returns
            it converted to the appropriate data-type.
        check_bool: Checks whether the input value is a valid boolean or boolean-equivalent and returns it converted to
            the appropriate data-type.
        check_string: Checks whether the input value is a valid string and returns it converted to the appropriate
            data-type. Converts all input values to strings if allow_string_conversion class attribute is True.
        check_none: Checks whether the input value is a valid NoneType or equivalent string and returns it converted
            to appropriate data-type.
    """

    def __init__(
            self,
            allow_bool: bool,
            allow_string: bool,
            allow_int: bool,
            allow_float: bool,
            allow_none: bool,
            allow_iterables: bool,
            allow_string_conversion: bool = False,
            string_options: Optional[list[str] | tuple[str] | set[str]] = None,
            string_force_lower: bool = False,
            number_lower_limit: Optional[int | float] = None,
            number_upper_limit: Optional[int | float] = None,
            parse_bool_equivalents: bool = False,
            parse_none_equivalents: bool = True,
            parse_number_strings: bool = True,
            iterable_output_type: Optional[Literal["tuple", "list", "tuple"]] = None,
    ) -> None:
        self.supported_iterables = {"set": set, "tuple": tuple, "list": list}

        # Checks some inputs, mostly for cases that can lead to implicit failures (incorrect class behavior, due to
        # logically well-formed, but semantically incorrect input).
        # If string-option-limiting is enabled checks the input for validity
        if string_options is not None:
            # If the options list is empty, this is treated as an unintentional equivalent of setting the variable to
            # None and an exception is raised.
            if len(string_options) < 1:
                custom_error_message = (
                    f"An empty string-option iterable detected when initializing ValueConverter class instance. "
                    f"Provide a tuple, list or set of string-datatype options or set string_options argument to None to"
                    f"disable limiting string-options."
                )
                raise ValueError(custom_error_message)

            # If any of the options is not a string, this will implicitly fail the validation of the string, which is
            # considered undesirable. Raises an exception to ask for the options to be strings
            for option in string_options:
                if not isinstance(option, str):
                    custom_error_message = (
                        f"Unsupported string-option {option} detected when initializing ValueConverter class instance. "
                        f"Provide a tuple, list or set of string-datatype options or set string_options argument to "
                        f"None to disable limiting string-options."
                    )
                    raise ValueError(custom_error_message)

        # Similarly, checks iterable_output_type for validity
        if iterable_output_type is not None and iterable_output_type not in self.supported_iterables.keys():
            custom_error_message = (
                f"Unsupported output iterable string-option {iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{self.supported_iterables.keys()}."
            )
            raise ValueError(custom_error_message)

        # Sets conversion / validation attributes
        self.allow_bool = allow_bool
        self.allow_string = allow_string
        self.allow_int = allow_int
        self.allow_float = allow_float
        self.allow_none = allow_none
        self.allow_iterables = allow_iterables
        self.allow_string_conversion = allow_string_conversion
        self.string_options = string_options
        self.string_force_lower = string_force_lower
        self.number_lower_limit = number_lower_limit
        self.number_upper_limit = number_upper_limit
        self.parse_bool_equivalents = parse_bool_equivalents
        self.parse_none_equivalents = parse_none_equivalents
        self.parse_number_strings = parse_number_strings
        self.iterable_output_type = iterable_output_type

        # Sets equivalent string sets for boolean and none variables. Note, the algorithm checking inputs against
        # these sets IS case-sensitive, so provide both upper and lower case strings if you want to cover all possible
        # options
        self.true_equivalents = {"True", "true", 1, "1", 1.0}
        self.false_equivalents = {"False", "false", 0, "0", 0.0}
        self.none_equivalents = {"None", "none", "Null", "null"}  # Uses set for efficiency

    def __repr__(self) -> str:
        """Gives the class a nicely formatted representation for print() and other functions designed to make use of
        repr calls.

        Returns:
            A nicely formatted string that includes the values of the key attributes of the class to represent the class
            object.
        """
        id_string = (
            f"ValueValidator(allow_int={self.allow_int}, allow_float={self.allow_float}, allow_bool={self.allow_bool}, "
            f"allow_none={self.allow_none}, allow_string={self.allow_string}, allow_iterables={self.allow_iterables}, "
            f"allow_string_conversion={self.allow_string_conversion}, string_options={self.string_options}, "
            f"string_force_lower={self.string_force_lower}, number_lower_limit={self.number_lower_limit}, "
            f"number_upper_limit={self.number_upper_limit}, parse-bool_equivalents={self.parse_bool_equivalents}, "
            f"parse_none_equivalents={self.parse_none_equivalents}, parse_number_strings={self.parse_number_strings}, "
            f"iterable_output_type={self.iterable_output_type}, true_equivalents={self.true_equivalents}, "
            f"false_equivalents={self.false_equivalents}, none_equivalents={self.none_equivalents}, "
            f"supported_iterables={self.supported_iterables})"
        )
        return id_string

    def convert_value(
            self, value: str | int | float | bool | None | list | tuple | set, multiprocessing: bool = False
    ) -> str | int | float | bool | None | list | tuple | set | AtaraxisError:
        """Validates the input value and, if needed, converts it to the correct datatype before returning to caller.

        This is the main function of the class that calls other class functions as necessary. It handles setting-up the
        validation procedure for both single variable inputs and iterables (iterates over iterables and calls validation
        subroutines on each iterable member value) and multiprocessing-related error wrapping.

        For each single value that passes validation, returns the value converted to the most preferred valid datatype
        in the order:  int > float > bool > none > string. For iterables that pass validation, first processes each
        value in the iterable using the procedure outlined above. Then, either converts the iterable to the preferred
        output type specified by iterable_output_type class attribute or to the same type as the input iterable if no
        preference is provided (class attribute is set to None).

        Args:
            value: The value to be validated. Currently, accepts strings, integers, floats, booleans and NoneTypes under
                all circumstances and shallow lists, tuples and sets if class allow_iterables attribute is True.
            multiprocessing: Use this flag to specify when this function is handed off to a parallel worker.
                In that case, the raised error messages will be packaged and returned as AtaraxisError class instance.
                This is required to properly handle tracebacks in multiprocessing environments. Defaults to False.

        Returns:
            The verified value or an iterable of verified values for iterable inputs converted to the most preferred
            valid datatype. For example, a string input value '14.15' will be returned as a float, if floats are
            allowed, as floats are preferred over strings.
            AtaraxisError class instance if an error is encountered and multiprocessing flag is set to True.

        Raises:
            TypeError: If the type of the value to be verified is not one of the supported types, including cases when
                it is a list, tuple or set when allow_iterables class attributes is set to False.
            ValueError: If validation fails for the input value or any value in the input iterable.
            RuntimeError: If the input value is a set, the class is configured to return a set as an output and the
                number of items in the output set does not match the number of items in the input set.
            Exception: If an unexpected error is encountered during runtime.
        """
        handled = False
        try:
            in_type = type(value).__name__  # Records the incoming value type. Only used for iterable processing

            # Unconditionally allows standard single python types to pass to the verification step. Also
            # allows most standard python iterables, if class attribute allow_iterables is True.
            if not isinstance(value, (str, int, float, bool, NoneType)) and not (
                    isinstance(value, (tuple, list, set)) and self.allow_iterables
            ):
                # Produces slightly different error messages for the case where iterables are allowed and when they are
                # not, but generates an error message whenever the input type is not an acceptable type given class
                # configuration (iterables allowed or not)
                if not self.allow_iterables:
                    custom_error_message = (
                        f"Expected a single integer, boolean, float, string or None input when converting value, but "
                        f"encountered a value of '{type(value).__name__}' type instead. To enable processing a "
                        f"tuple, list or set inputs, set allow_list class attribute to True."
                    )
                else:
                    custom_error_message = (
                        f"Expected a single integer, boolean, float, string or None input or a tuple, list, "
                        f"or set iterable of such inputs when converting value, but encountered a value of "
                        f"'{type(value).__name__}' type instead."
                    )
                handled = True
                raise TypeError(custom_error_message)

            # Converts all inputs to iterable (tuple) format to reuse the same processing code for all supported
            # input types (also for efficiency)
            value = tuple(ensure_list(input_item=value))

            # Uses list comprehension to run the validation function on each value in the tuple generated above
            try:
                out_value = [self.validate_value(var) for var in value]
            except ValueError as e:
                # If validation fails, augments the message with a slightly different head-string to indicate if the
                # failed value was a single-value input or part of an iterable
                if in_type in self.supported_iterables.keys():
                    custom_error_message = (
                        f"Unable to validate at least one value in the input iterable {value} when converting value."
                    )
                else:
                    custom_error_message = f"Unable to validate the input value when converting value."
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            # Has a slightly different return procedure for iterable and non-iterable input type
            if in_type in self.supported_iterables.keys():
                # If the class is configured to return specific iterable types for iterables, converts the output
                # list to the requested datatype
                if self.iterable_output_type is not None:
                    value_to_return = self.supported_iterables[self.iterable_output_type](out_value)
                # Otherwise, converts the output list to the same datatype as the input iterable
                else:
                    value_to_return = self.supported_iterables[in_type](out_value)

                # If the input is a 'set' iterable, it is possible that the validation converts two or more input
                # values in the set to duplicates. Since sets do not allow duplicates, this would eliminate the
                # excessive duplicated values when the validated list is converted back into set. This situation is
                # caught by this exception block
                if in_type == "set" and len(value_to_return) != len(value):
                    custom_error_message = (
                        f"The number of values ({len(value_to_return)}) in the output set does not match the number of "
                        f"values in the input set ({len(value)}), when converting value. This is likely due to the "
                        f"validation process producing one or more duplicated values."
                    )
                    raise RuntimeError(custom_error_message)
                else:
                    return value_to_return  # Returns the iterable back to caller
            else:
                # If the input is a single value, pops it out of the output list and returns it to caller
                return out_value.pop(0)

        except Exception as e:
            if not handled:
                custom_error_message = f"Unexpected error when converting value."
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error

    def validate_value(self, value_to_validate: Any) -> int | float | bool | None | str:
        """Checks the input value against all value types supported by the validator class.

        Evaluates the input value against class validation conditions to determine whether it conforms to the
        range of datatypes and values supported by the validator class. If validation fails, raises a ValueError
        exception. Otherwise, returns the most preferred successfully validated value type (see below).

        When the validator class is configured to allow multiple output value types, it uses the following order of
        preference when returning the variable: int > float > bool > none > string. This sequence is generally optimal
        for most intended use cases, but can be implicitly overriden with various class attributes modifying each
        verification step (for example, it is possible to disable numeric strings from being recognized as integers and
        floats, ensuring strings are returned as strings even if they are number-convertible).

        Note, this function is intended to be called by the convert_value() function and, as such, it does not contain
        multiprocessing error handling capabilities. It expects the caller function to handle the necessary error
        message wrapping to support error messaging during multiprocessing execution.

        Args:
            value_to_validate: The value to be validated.

        Returns:
            The value converted to the most preferred datatype that passed validation.

        Raises:
            ValueError: If the value cannot be validated as any of the supported datatypes.
        """
        # If allowed, parses the variable as a number.
        if self.allow_int | self.allow_float:
            num_var = self.check_num(value_to_check=value_to_validate)

            # If numeric lower range limiting is enabled, invalidates the number if it is less than the lower limit
            if self.number_lower_limit is not None and num_var is not None and num_var < self.number_lower_limit:
                num_var = None

            # If numeric upper range limit is enabled, invalidates the number if it is higher than the lower limit
            elif self.number_upper_limit is not None and num_var is not None and num_var > self.number_upper_limit:
                num_var = None
        else:
            num_var = None

        # If allowed, parses the variable as a boolean
        if self.allow_bool:
            bool_var = self.check_bool(value_to_check=value_to_validate)
        else:
            bool_var = None

        # If allowed, checks if the variable is None. Note, unlike other checks, here returned None is
        # considered successful and returned False is considered a failure.
        if self.allow_none:
            none_var = self.check_none(value_to_check=value_to_validate)
        else:
            none_var = False

        # If allowed, parses the variable as a string
        if self.allow_string:
            string_var = self.check_string(value_to_check=value_to_validate)
        else:
            string_var = None

        # If at least one validation step returns a passing value, selects the first valid value in the order
        # num (int > float) > bool > none > string and returns it to caller
        # The primary reason for numeric processing being first is that
        # booleans can be represented with integer 0 and 1, so processing numbers first ensures that integers are not
        # converted to booleans when both booleans and integers are valid value types.
        if not isinstance(num_var, NoneType):
            return num_var
        elif not isinstance(bool_var, NoneType):
            return bool_var
        elif not isinstance(none_var, bool):
            return none_var
        elif not isinstance(string_var, NoneType):
            return string_var
        # Otherwise, raises an exception to communicate the failed value, type and validator parameters that jointly
        # led to validation failure
        else:
            custom_error_message = (
                f"Failed to validate the value '{value_to_validate}' of type '{type(value_to_validate).__name__}'. "
                f"Verification parameters: allow_int={self.allow_int}, allow_float={self.allow_float}, "
                f"allow_bool={self.allow_bool}, allow_none={self.allow_none}, allow_string={self.allow_string}, "
                f"allow_string_conversion={self.allow_string_conversion}, string_options={self.string_options}, "
                f"number_lower_limit={self.number_lower_limit}, number_upper_limit={self.number_upper_limit}, "
                f"parse-bool_equivalents={self.parse_bool_equivalents}, "
                f"parse_none_equivalents={self.parse_none_equivalents}, "
                f"parse_number_strings={self.parse_number_strings}."
            )
            raise ValueError(custom_error_message)

    def check_num(self, value_to_check: Any) -> int | float | None:
        """Checks if the input value is an integer or float.

        If possible, tries to convert the input into an integer to potentially optimize future computations, as
        integers tend to use less memory than floats in many use cases. If this is not possible, for example if
        conversion is not possible without rounding-associated data loss, returns the float with the same precision as
        the input. Tries to parse incoming strings as numbers (integers or floats).

        Note, this is an internal function that is intended to be used exclusively by other class functions.

        Args:
            value_to_check: Any object that may be an integer, float or number-parsable string variable convertible to
                integer or float.

        Returns:
            Integer or float value, depending on whether class attributes allow_int and allow_float are set to True
            (prioritizes returning integers over floats where possible to conserve memory use).
            None, if verification fails.
        """

        # Python converts boolean True to 1 and boolean False to 0. For the scope of this project such
        # equivalence is considered undesirable, hence an attempt to validate a boolean type automatically results in
        # failure.
        if isinstance(value_to_check, bool):
            return None

        # Attempts to convert input strings to a numeric value if parsing numeric strings is allowed
        if isinstance(value_to_check, str) and self.parse_number_strings:
            try:
                # Converts strings to float, as integers can always be represented as floats, but the opposite is not
                # always possible
                value_to_check = float(value_to_check)
            except ValueError:
                return None  # If the string is not number-convertible, validation fails with None return

        # Checks if the numeric value (either original or converted from string) can be represented as an integer or
        # float
        if isinstance(value_to_check, (int, float)):
            # Prefers integers, so first checks if integers are allowed and the value can be converted to / already is
            # an integer and, if so, returns an integer
            if value_to_check.is_integer() and self.allow_int:
                return int(value_to_check)
            # Otherwise, if floats are allowed, returns the value as a float
            elif self.allow_float:
                return float(value_to_check)

        return None  # If the function reaches this point, validation is failed, so returns None

    def check_string(self, value_to_check: Any) -> str | None:
        """Checks if the input value is a string.

        If option-limiting is enabled, evaluates the input string against the provided Iterable of options.
        If the string is evaluated against options, it is converted to lower-case to support case-insensitive
        validation. If the value passes validation, the output string is not returned as lower_case, unless
        class string_force_lower attribute is set to True.

        Note, this is an internal function that is intended to be used exclusively by other class functions.

        Args:
            value_to_check: Any object that may be a string value.

        Returns:
            String value, converted to lower case if requested, if the value is a valid string and (optionally) matches
            one of the valid options if option-limiting is enabled.
            Also returns a string value for any input value type, if class attribute allow_string_conversion is True
            (since any valid input value is, by default, string-convertible).
            None, if verification fails.
        """

        # Ensures that the input variable is a string, otherwise returns None to indicate check failure. If the variable
        # is originally not a string, but string-conversions are allowed, attempts to convert it to string, but returns
        # None if the conversion fails (unlikely)
        if not isinstance(value_to_check, str) and not self.allow_string_conversion:
            return None
        else:
            try:
                value_to_check = str(value_to_check)
            except Exception:
                return None

        # If needed, converts the checked value to lower case. This is done either if the validator is configured to
        # convert strings to lower case or if it is configured to evaluate the string against an iterable of options.
        # In the latter case, the value can still be returned as non-lower-converted string, depending on the
        # 'string_force_lower' attribute setting.
        value_lower = value_to_check.lower() if self.string_force_lower or self.string_options else value_to_check

        # If option-limiting is enabled, validates the value against the iterable of options
        if self.string_options:
            # Converts options to lower case as an extra compatibility improvement step (potentially avoids user=input
            # errors)
            option_list_lower = [option.lower() for option in self.string_options]

            # Checks if value is in the options list
            if value_lower in option_list_lower:
                # If the validator is configured to convert strings to lower case, returns lower-case string
                if self.string_force_lower:
                    return value_lower
                # Otherwise returns the original input string without alteration
                else:
                    return value_to_check
            else:
                # If the value is not in the options list or if the options list is empty, returns None to indicate
                # check failure
                return None

        # If option-limiting is not enabled, returns the string value
        return value_lower

    def check_bool(self, value_to_check: Any) -> bool | None:
        """Checks if the input value is a boolean True/False or a boolean-equivalent value.

        Accepts string 'True/False', 'true/false' or string or integer 1/0 values as boolean-equivalents, if the
        class is configured to parse boolean-equivalents.

        Note, this is an internal function that is intended to be used exclusively by other class functions.

        Args:
            value_to_check: Any object that may be a boolean value or an equivalent integer/string.

        Returns:
            Boolean True or False, if the value is a boolean or an equivalent integer/string and parsing
            boolean-equivalents is enabled.
            None, if verification fails.
        """

        # If the input is a boolean type returns it back to caller unchanged
        if isinstance(value_to_check, bool):
            return value_to_check

        # Otherwise, if the value is a boolean-equivalent string or number and parsing boolean-equivalents is allowed
        # converts it to boolean True or False and returns it to caller
        if self.parse_bool_equivalents and isinstance(value_to_check, (str, int)):
            if value_to_check in self.true_equivalents:
                return True
            elif value_to_check in self.false_equivalents:
                return False

        # If the function reaches this point the value is not a boolean or parsable boolean-equivalent, so returns None
        # to indicate check failure
        return None

    def check_none(self, value_to_check: Any) -> None | bool:
        """Checks if the input value is a None or a None-equivalent value.

        Accepts None-equivalent strings and converts them to pythonic None if the class is configured to parse
        None-equivalents. None-equivalent strings are: 'None', 'none', 'null', 'Null'.

        Note, this is an internal function that is intended to be used exclusively by other class functions.

        Args:
            value_to_check: Any object that may be a pythonic None value or an equivalent string.

        Returns:
            None, if the value is None or a None-equivalent string and parsing None-strings is enabled.
            Boolean False if verification fails.
        """

        # If the input is pythonic None, returns None
        if value_to_check is None:
            return None

        # If the input is a pythonic-None-equivalent string and the validator is configured to parse none-equivalent
        # strings, returns None
        elif value_to_check in self.none_equivalents and self.parse_none_equivalents:
            return None

        # Otherwise, returns False (check failed). Since None is reserved as the passing result, uses False for this
        # function
        else:
            return False


@dataclass
class YamlConfig:
    """This class functions as general method repository for all Ataraxis configuration classes.

    It stores methods that are frequently re-used in various runtime configuration subclasses used across Ataraxis
    pipelines. Note, during inheritance, polymorphism is used to redeclare some of the methods listed here to contain
    subclass-specific runtimes, such as create_instance and define_used_runtime_parameters.

    See specific children subclasses that inherit this class for docstrings on the purpose and setup of each config
    subclass.

    Notes:
        For developers. Generally, you should not need to use this class directly or edit any of its parameters. If
        you are writing new config classes, make sure they inherit the instance of this base class.

    Methods:
        remove_unused_parameters: Removes parameters not used by runtime (according to runtime_parameters dict) from
                the input class instance.
        write_config_to_file: Writes the class instance to a.yaml file.
        read_config_from_file: Reads the class instance from a.yaml file, validates inputs and generates a configured
            class instance that can be passed to runtime/pipeline parameter controller class.
    """

    def save_config_as_yaml(self, config_path: str) -> None:
        """Converts the class instance to a dictionary and saves it as a .yaml file.

        This method is used to store the software axis configuration parameters between runtimes by dumping the data
        into an editable .yaml file. As such, this can also be used to edit the parameters between runtimes, similar to
        how many other configuration files work.

        Args:
            config_path: The path to the .yaml file to write. If the file does not exist, it will be created, alongside
                any missing directory nodes. If it exists, it will be overwritten (re-created).

        Raises:
            ValueError: If the output path does not point to a file with a '.yaml' or '.yml' extension.
        """

        # Defines YAML formatting options. The purpose of these settings is to make yaml blocks more readable when
        # being edited offline.
        yaml_formatting = {
            "default_style": "",  # Use single or double quotes for scalars as needed
            "default_flow_style": False,  # Use block style for mappings
            "indent": 10,  # Number of spaces for indentation
            "width": 200,  # Maximum line width before wrapping
            "explicit_start": True,  # Mark the beginning of the document with ___
            "explicit_end": True,  # Mark the end of the document with ___
            "sort_keys": False,  # Preserves the order of key as written by creators
        }

        # Ensures that output file path points to a .yaml (or .yml) file
        if not config_path.endswith(".yaml") and not config_path.endswith(".yml"):
            custom_error_message = (
                f"Invalid file path provided when attempting to write the axis configuration parameters to a yaml "
                f"file. Expected a path ending in the '.yaml' or '.yml' extension, but encountered {config_path}. "
                f"Provide a path that uses the correct extension."
            )
            raise ValueError(format_exception(custom_error_message))

        # Ensures that the output directory exists. This is helpful when this method is invoked for the first time for
        # a given axis and runtime combination which may not have the necessary directory nodes available.
        self._ensure_directory_exists(config_path)

        # Writes the data to a .yaml file using custom formatting defined at the top of this method.
        with open(config_path, "w") as yaml_file:
            yaml.dump(asdict(self), yaml_file, **yaml_formatting)

    @classmethod
    def load_config_from_yaml(cls, config_path: str) -> "ZaberAxisConfig":
        """Loads software parameter values from the .yaml storage file and uses them to generate an instance of the
        config class.

        This method is designed to load the parameters saved during a previous runtime to configure the next runtime(s).

        Args:
            config_path: The path to the .yaml file to read the parameter values from.

        Returns:
            A new ZaberAxisConfig class instance created using the data read from the .yaml file.

        Raises:
            ValueError: If the provided file path does not point to a .yaml or .yml file.
        """

        # Ensures that config_path points to a .yaml / .yml file.
        if not config_path.endswith(".yaml") and not config_path.endswith(".yml"):
            custom_error_message = (
                f"Invalid file path provided when attempting to read software axis configuration parameters from a "
                f".yaml storage file. Expected a path ending in the '.yaml' or '.yml' extension, but encountered "
                f"{config_path}. Provide a path that uses the correct extension."
            )
            raise ValueError(format_exception(custom_error_message))

        # Opens and reads the .yaml file. Note, safe_load may not work for reading python tuples, so it is advised
        # to avoid using tuple in configuration files.
        with open(config_path, "r") as yml_file:
            data = yaml.safe_load(yml_file)

        # Converts the imported data to a python dictionary.
        config_dict: dict = dict(data)

        # Uses the imported dictionary to instantiate a new class instance and returns it to caller.
        return cls(**config_dict)

    @staticmethod
    def _ensure_directory_exists(path: str) -> None:
        """Determines if the directory portion of the input path exists and, if not, creates it.

        When the input path ends with a .extension (indicating this a file path), the file portion is ignored and
        only the directory path is evaluated.

        Args:
            path: The string-path to be processed. Should use os-defined delimiters, as os.path.splitext() is used to
                decompose the path into nodes.
        """
        # Checks if the path has an extension
        _, ext = os.path.splitext(path)

        if ext:
            # If the path has an extension, it is considered a file path. Then, extracts the directory part of the path.
            directory = os.path.dirname(path)
        else:
            # If the path doesn't have an extension, it is considered a directory path.
            directory = path

        # Checks if the directory hierarchy exists.
        if not os.path.exists(directory):
            # If the directory hierarchy doesn't exist, creates it.
            os.makedirs(directory)

    @classmethod
    def create_instance(cls) -> tuple["YamlConfig", "YamlConfig"]:
        """This is a placeholder method.

        Due to python Polymorphism, it will be overridden and replaced by the
        subclass that inherits from base config class. It is here just to avoid annoying IDE errors.

        See the docstrings inside the runtime_configs library for the pipeline you want to configure if you need
        help setting, updating or using this method.
        """
        return YamlConfig(), YamlConfig()

    @classmethod
    def define_used_runtime_parameters(cls, runtime: str) -> dict:
        """This is a placeholder method.

        Due to python Polymorphism, it will be overridden and replaced by the
        subclass that inherits from base config class. It is here just to avoid annoying IDE errors.

        See the docstrings inside the runtime_configs library for the pipeline you want to configure if you need
        help setting, updating or using this method.
        """
        return dict()

    @classmethod
    def remove_unused_parameters(
            cls,
            class_dict: dict,
            parameter_dict: dict,
    ) -> dict:
        """Removes all elements of the default class instance that are not relevant for the currently active runtime.

        This operation is carried out on the default class instance that includes all parameters.
        By removing unused parameters, the algorithm improves user experience and reduces the possibility
        of incorrect configuration.
        It works on a dictionary-converted instance of the class during both .yaml writing and reading method execution.

        If define_used_runtime_parameters and the main class are configured correctly, this method should automatically
        work for all runtimes. As such, even devs should not really have a reason to modify this method.

        Args:
            class_dict: The hierarchical dictionary to be written as .yaml file. By design, this is the default
                instance of the class that contains all possible parameters.
            parameter_dict: A dictionary that stores lists of used parameter ID's for each configurable function.

        Returns:
            The truncated version of the hierarchical dictionary with all unused parameters removed.

        Raises:
            Exception: If one of the called functions returns an error or if an unexpected error is encountered.

        """
        handled = False
        try:
            # Unpacks the paths to parameters using the general hierarchical dictionary crawler.
            # For this dictionary, returns a list of parameter paths, each being a one-element list (due to 1-element
            # hierarchy)
            try:
                param_paths = extract_nested_variable_paths(parameter_dict, delimiter=".", multiprocessing=False)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = f"Unable to extract parameter paths from parameter_dict."

                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            used_parameters = []

            # Loops over each path and uses it to retrieve the matching parameter value
            for delimited_path in param_paths:
                try:
                    used_parameter_list = read_nested_value(
                        source_dict=parameter_dict,
                        variable_path=delimited_path,
                        delimiter=".",
                    )
                except Exception as e:
                    # Provides a custom error message
                    custom_error_message = (
                        f"Unable to read the nested parameter value from runtime parameter dictionary "
                        f"using path {delimited_path}."
                    )

                    handled = True
                    augment_exception_message(e=e, additional_message=custom_error_message)
                    raise

                # Extends ALL parameter IDs into a single mega-list
                used_parameters.extend(used_parameter_list)

            # Next, extracts the paths to all available parameters inside class_dict
            try:
                class_param_paths = extract_nested_variable_paths(class_dict, delimiter=".", multiprocessing=False)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = f"Unable to extract parameter paths from class default instance dictionary."

                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            # Loops over each extracted path and retrieves the last variable in the list, which is necessarily the
            # parameter name. Checks the name against the list of all used parameter and, if the parameter is not used,
            # removes the parameter from dictionary
            for path in class_param_paths:
                paths = path.split(".")
                if paths[-1] not in used_parameters:
                    try:
                        class_dict = delete_nested_value(target_dict=class_dict, path_nodes=path)
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to remove the specified path {path} from " f"class default instance dictionary."
                        )

                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

            # Returns truncated class dictionary to caller
            return class_dict
        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = f"Unexpected error when removing unused default class instance parameters."
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise

    @classmethod
    def write_config_file(cls, output_path: str, runtime: str) -> None:
        """Instantiates, presets and writes a runtime-specific instance of the class as a .yaml file.

        This method combines all other methods necessary to generate a default class instance and write it to output
        directory as a .yaml file.
        Specifically, it instantiates the default class instance, determines which parameters are used by the active
        runtime, removes unused parameters and then saves the truncated config file to the provided output_directory.

        Notes:
            For developers. This method should be class-agnostic and work for any generally formatted Ataraxis Config
            class.

        Args:
            output_path: The output path for the .yaml file to be created. Note, has to include .yaml
                extension
            runtime: The ID of the currently active runtime. This is used to remove unused parameters from the default
                dictionary. This should be set via pipeline's argparse module.

        Returns:
            Does not explicitly return anything, but generates a .yaml file using the output path.

        Raises:
            Exception: If an unexpected error is encountered or if any of the used subroutines encounter an error.
        """
        handled = False
        try:
            # Instantiates the class and converts it to nested dictionary
            (
                class_instance,
                _,
            ) = cls.create_instance()  # Ignores the validator for this method
            # noinspection PyDataclass
            class_dict = asdict(class_instance)

            # Adds help hint section to the dictionary
            class_dict["addendum"] = {
                "Help": "Use the README.md file or the API documentation available through the GitHub repository "
                        "(https://github.com/Inkaros/Ataraxis_Data_Processing) if you need help editing this file"
            }

            try:
                # Uses runtime to obtain creator-defined list of parameters used by that specific pipeline
                used_parameters = cls.define_used_runtime_parameters(runtime)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to parsed used parameters for {runtime} while writing {cls.__name__} class as .yaml."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            try:
                # Pops unused parameters available in the default class dictionary and its validator mirror
                # This truncates the config files generate by this method to only include used parameters
                class_dict = cls.remove_unused_parameters(class_dict, used_parameters)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to remove unused parameters for {runtime} from default {cls.__name__} class instance "
                    f"while writing it as yaml."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            try:
                # Writes config file to .yaml file inside configs output subfolder
                write_dict_to_yaml(file_path=output_path, dict_to_write=class_dict)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = f"Unable to write configured {cls.__name__} class instance as .yaml file."

                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = (
                    f"Unexpected error when attempting to instantiate and write {cls.__name__} "
                    f"config class as yaml. file"
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise

    @classmethod
    def read_config_file(cls, input_path: str, runtime: str) -> Any:
        """Reads and validates a user-configured config .yaml file and uses it to set runtime parameters.

        This method combines all other methods necessary to read a user-configured config .yaml file (after it has
        been saved via write_config_file() method).

        To do so, instantiates the default class instance and its mirror validator instance and loads the config file.
        Then, compares each parameter in the default instance to the parameter loaded from the config file, using
        matching validator to ensure that the parameter is set to an acceptable value. Sets all unused parameters to
        None. Once the check is complete, stores all set parameters as a dataclass object instance to be used by
        runtime functions.

        Notes:
            For developers. This method should be class-agnostic and work for any generally formatted Ataraxis Config
            class.

        Args:
            input_path: The input path that points to a .yaml file to read. Note, has to include .yaml
                extension.
            runtime: The ID of the currently active runtime. This is used to remove unused parameters from the default
                dictionary. This should be set via pipeline's argparse module.

        Returns:
            A configured instance of the class to be passed as runtime argument.

        Raises:
            Exception: If an unexpected error is encountered or if any of the used subroutines encounter an error.
        """
        handled = False
        try:
            # Instantiates the default class instance and its mirror validator and converts them to dictionaries
            class_instance, validator_class = cls.create_instance()
            # noinspection PyDataclass
            class_dict = asdict(class_instance)
            # noinspection PyDataclass
            validator_dict = asdict(validator_class)

            try:
                # Uses runtime to obtain creator-defined list of parameters used by that specific pipeline
                used_parameters = cls.define_used_runtime_parameters(runtime)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to parse used parameters for {runtime} while reading {cls.__name__} class from .yaml file."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            try:
                # Pops unused parameters from the validator (but not default! dictionary)
                validator_trimmed_dict = cls.remove_unused_parameters(validator_dict, used_parameters)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to remove unused parameters for {runtime} from validator {cls.__name__} class instance "
                    f"while reading it from .yaml file."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            try:
                # Imports the yaml file as a dictionary
                imported_class_dict = read_dict_from_yaml(file_path=input_path)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = f"Unable to read user-configured {cls.__name__} class instance from .yaml file."
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            # At this point, there are 3 dictionaries: default (full-size),
            # imported (trimmed) and validator (also trimmed!)

            # Parses ALL available parameter paths from main dict
            try:
                all_parameter_paths = extract_nested_variable_paths(target_dict=class_dict)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to extracted parameter paths from default instance dictionary while reading "
                    f"{cls.__name__} class from .yaml file."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            # Parses used parameter dict (NOTE, instead of parameters, it returns lists of parameters available for
            # each lowest level dictionary section in a given hierarchy). This makes the procedure class-agnostic.
            try:
                used_param_paths = extract_nested_variable_paths(used_parameters)
            except Exception as e:
                # Provides a custom error message
                custom_error_message = (
                    f"Unable to extract used parameter paths from parameter_dict while reading "
                    f"{cls.__name__} class from .yaml file."
                )
                handled = True
                augment_exception_message(e=e, additional_message=custom_error_message)
                raise

            # Loops over each available parameter path and attempts to find it in the trimmed validator dictionary
            for param_path in all_parameter_paths:
                # Concatenates the parameter path so that it can be used to read adn write nested dictionaries
                parameter_path_string = ".".join(param_path)

                # Checks if each default class parameter is used by the current runtime.

                # First, loops over each used_parameters list path, imports the list and checks if the evaluated
                # default class parameter is found in any list
                param_found = False
                for used_param_path in used_param_paths:
                    used_param_path = ".".join(used_param_path)

                    try:
                        used_param_list = read_nested_value(
                            source_dict=used_parameters,
                            variable_path=used_param_path,
                            delimiter=".",
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to read used parameter list from {used_param_path} while "
                            f"reading {cls.__name__} class from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

                    if param_path[-1] in used_param_list:
                        param_found = True
                        break

                # If the parameter is not found in any list, sets the parameter inside the class dictionary to None,
                # which disables the use of the parameter
                if not param_found:
                    try:
                        write_nested_value(
                            target_dict=class_dict,
                            variable_path=parameter_path_string,
                            value=None,
                            delimiter=".",
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to write {cls.__name__} class parameter to path "
                            f"{parameter_path_string} while reading the class from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

                    # Also adds the None parameter to the imported dictionary.
                    # This needs to be done due to how validators use readout toggles: some variables are only
                    # imported if their toggle (some other variable) is set to some particular value. The toggles are
                    # expected to be found in the same dictionary as the value that is to be read. Hence, if some
                    # value is not present in the main config, but it is used as a toggle for some other variable that
                    # is present, it needs to be re-introduced into important config as a None value.
                    try:
                        write_nested_value(
                            target_dict=imported_class_dict,
                            variable_path=parameter_path_string,
                            value=None,
                            delimiter=".",
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to write {cls.__name__} class parameter to imported class instance path "
                            f"{parameter_path_string} while reading the class from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

                # Otherwise, uses a matching validator class to read and validate the parameter
                else:
                    try:
                        # Extracts the matching validator instance from the validator dictionary
                        validator = read_nested_value(
                            source_dict=validator_trimmed_dict,
                            variable_path=parameter_path_string,
                            delimiter=".",
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to extract {cls.__name__} class validator from path "
                            f"{parameter_path_string} while reading it from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

                    try:
                        # Reads and validates the parameter value from imported dictionary
                        result = validator.read_value(
                            source_dict=imported_class_dict,
                            dict_name=f"{cls.__name__}",
                            variable_path=parameter_path_string,
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to Validate {cls.__name__} class parameter from path "
                            f"{parameter_path_string} while reading it from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

                    # If the validation succeeds, sets the parameter value to the validated value
                    try:
                        write_nested_value(
                            target_dict=class_dict,
                            variable_path=parameter_path_string,
                            value=result,
                            delimiter=".",
                        )
                    except Exception as e:
                        # Provides a custom error message
                        custom_error_message = (
                            f"Unable to write {cls.__name__} class parameter from path "
                            f"{parameter_path_string}while reading it from .yaml file."
                        )
                        handled = True
                        augment_exception_message(e=e, additional_message=custom_error_message)
                        raise

            # Converts the dictionary back into the dataclass format using **kwarg assignment and returns it to caller
            # noinspection PyArgumentList
            return cls(**class_dict)

        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = (
                    f"Unexpected error when importing and validating {cls.__name__} config from .yaml file."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise


class SharedMemoryArray:
    """A wrapper around an n-dimensional numpy array object that exposes methods for accessing the array buffer from
    multiple processes.

    This class is designed to compliment the Queue-based method for sharing data between multiple python processes.
    Similar to Queues, this class instantiates a shared memory buffer, to which all process-specific instance of this
    class link when their 'connect' property is called. Unlike Queue, however, this shared memory buffer is static
    post-initialization and represents a numpy array with all associated limitations (fixed datatypes, static size,
    etc.).

    This class should only be instantiated inside the main process via the create_array() method. Do not attempt to
    instantiate the class manually. All children processes should get an instance of this class as an argument and
    use the connect() method to connect to the buffer created by the founder instance inside the main scope.

    Notes:
        Shared memory objects are garbage-collected differently depending on the host-platform OS. On Windows-based
        systems, garbage collection is handed off to the OS and cannot be enforced manually. On Unix systems, the
        buffer can be garbage-collected via appropriate de-allocation commands.

        All data accessors from this class use multiprocessing Lock instance to ensure process- and thread-safety. This
        make this class less optimized for use-cases that rely on multiple processes simultaneously reading the same
        data for increased performance. In this case, it is advised to use a custom implementation of the shared
        memory system.

    Args:
        name: The descriptive name to use for the shared memory array. Names are used by the host system to identify
            shared memory objects and, therefore, have to be unique.
        shape: The shape of the numpy array, for which the shared memory buffer would be instantiated. Note, the shape
            cannot be changed post-instantiation.
        datatype: The datatype to be used by the numpy array. Note, the datatype cannot be changed post-instantiation.
        buffer: The memory buffer shared between all instances of this class across all processes (and threads).

    Attributes:
        _name: The descriptive name of the shared memory array. The name is sued to connect to the same shared memory
            buffer from different processes.
        _shape: The shape of the numpy array that uses the shared memory buffer. This is used to properly format the
            data available through the buffer.
        _datatype: The datatype of the numpy array that uses the shared memory buffer. This is also used to properly
            format the data available through the buffer.
        _buffer: The shared memory buffer that stores the array data. Has to be connected to vai connect() method
            before the class can be used.
        _lock: A Lock object used to ensure only one process is allowed to access (read or write) the array data at any
            point in time.
        _array: The inner object used to store the connected shared numpy array.
        _is_connected: A flag that tracks whether the shared memory array manged by this class has been connected to.
            This is a prerequisite for most other methods of the class to work.
    """

    def __init__(
            self,
            name: str,
            shape: tuple,
            datatype: np.dtype,
            buffer: Optional[SharedMemory],
    ):
        self._name: str = name
        self._shape: tuple = shape
        self._datatype: np.dtype = datatype
        self._buffer: Optional[SharedMemory] = buffer
        self._lock = Lock()
        self._array: Optional[np.ndarray] = None
        self._is_connected: bool = False

    @classmethod
    def create_array(cls, name: str, prototype: np.ndarray) -> "SharedMemoryArray":
        """Uses the input prototype numpy array to create an instance of this class.

        Specifically, this method first creates a shared bytes buffer that is sufficiently large to hold the data of the
        prototype array and then uses it to create a new numpy array with the same shape and datatype as the prototype.
        Subsequently, it copies all data from the prototype aray to the new shared memory array, enabling to access and
        manipulate the data from different processes (using returned class instance methods).

        Notes:
            This method should only be used once, when the array is first created in the root (main) process. All
            child processes should use the connect() method to connect to an existing array.

        Args:
            name: The name to give to the created SharedMemory object. Note, this name has to be unique across all
                scopes using the array.
            prototype: The numpy ndarray instance to serve as the prototype for the created shared memory object.

        Returns:
            The instance of the SharedMemoryArray class. This class exposes methods that allow connecting to the shared
            memory aray from different processes and thread-safe methods for reading and writing data to the array.

        Raises:
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists.
        """

        # Creates shared memory object. This process will raise a FileExistsError if an object with this name already
        # exists. The shared memory object is used as a buffer to store the array data.
        buffer = SharedMemory(create=True, size=prototype.nbytes, name=name)

        # Instantiates a numpy array using the shared memory buffer and copies prototype array data into the shared
        # array instance
        shared_arr = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
        shared_arr[:] = prototype[:]

        # Packages the data necessary to connect to the shared array into the class object.
        shared_memory_array = cls(
            name=name,
            shape=shared_arr.shape,
            datatype=shared_arr.dtype,
            buffer=buffer,
        )

        # Connects the internal array of the class object to the shared memory buffer.
        shared_memory_array.connect()

        # Returns the instantiated and connected class object to caller.
        return shared_memory_array

    def connect(self):
        """Connects to the shared memory buffer that stores the array data, allowing to manipulate access and manipulate
         the data through this class.

        This method should be called once for each process that receives an instance of this class as input, before any
        other method of this class. This method is called automatically as part of the create_array() method runtime for
        the founding array.
        """
        self._buffer = SharedMemory(name=self._name)  # Connects to the buffer
        # Connects to the buffer using a numpy array
        self._array = np.ndarray(shape=self._shape, dtype=self._datatype, buffer=self._buffer.buf)
        self._is_connected = True  # Sets the connection flag

    def disconnect(self):
        """Disconnects the class from the shared memory buffer.

        This method should be called whenever the process no longer requires shared buffer access.
        """
        if self._is_connected:
            self._buffer.close()
            self._is_connected = False

    def read_data(self, index: int | slice) -> np.ndarray:
        """Reads data from the shared memory array at the specified index or slice.

        Args:
            index: The integer index to read a specific value or slice to read multiple values from the underlying
                shared numpy array.

        Returns:
            The data at the specified index or slice as a numpy array. The data will use the same datatype as the
            source array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
            ValueError: If the input index or slice is invalid.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot read data as the class is not connected to a shared memory array. Use connect() method to "
                "connect to the shared memory array."
            )
            raise RuntimeError(format_exception(custom_error_message))

        with self._lock:
            try:
                return np.array(self._array[index])
            except IndexError:
                custom_error_message = (
                    "Invalid index or slice when attempting to read the data from shared memory array."
                )
                raise ValueError(format_exception(custom_error_message))

    def write_data(self, index: int | slice, data: np.ndarray) -> None:
        """Writes data to the shared memory array at the specified index or slice.

        Args:
            index: The index or slice to write data to.
            data: The data to write to the shared memory array. Must be a numpy array with the same datatype as the
                shared memory array bound by the class.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
            ValueError: If the input data is not a numpy array, if the datatype of the input data does not match the
                datatype of the shared memory array, or if the data cannot fit inside the shared memory array at the
                specified index or slice.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot write data as the class is not connected to a shared memory array. Use connect() method to "
                "connect to the shared memory array."
            )
            raise RuntimeError(format_exception(custom_error_message))

        if not isinstance(data, np.ndarray):
            custom_error_message = "Input data must be a numpy array."
            raise ValueError(format_exception(custom_error_message))

        if data.dtype != self._datatype:
            custom_error_message = (
                f"Input data must have the same datatype as the shared memory array: {self._datatype}."
            )
            raise ValueError(format_exception(custom_error_message))

        with self._lock:
            try:
                self._array[index] = data
            except ValueError:
                custom_error_message = (
                    "Input data cannot fit inside the shared memory array at the specified index or slice."
                )
                raise ValueError(format_exception(custom_error_message))

    @property
    def datatype(self) -> np.dtype:
        """Returns the datatype of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve array datatype as the class is not connected to a shared memory array. Use connect() "
                "method to connect to the shared memory array."
            )
            raise RuntimeError(format_exception(custom_error_message))
        return self._datatype

    @property
    def name(self) -> str:
        """Returns the name of the shared memory buffer.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve shared memory buffer name as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            raise RuntimeError(format_exception(custom_error_message))
        return self._name

    @property
    def shape(self) -> tuple:
        """Returns the shape of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve shared memory array shape as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            raise RuntimeError(format_exception(custom_error_message))
        return self._shape

    @property
    def is_connected(self) -> bool:
        """Returns True if the shared memory array is connected to the shared buffer.

        Connection to the shared buffer is required from most class methods to work.
        """
        return self._is_connected
