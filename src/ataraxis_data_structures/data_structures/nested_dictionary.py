import copy
import sys
from types import NoneType
from typing import Any, Literal, Optional

import numpy as np
from ataraxis_base_utilities import console


class NestedDictionary:
    """Wraps a nested (hierarchical) python dictionary and provides methods for manipulating its values.

    This class is primarily designed to abstract working with nested dictionaries without a-priori knowledge of
    dictionary layout. For example, when using data logging methods that produce nested .json or .yml files that can be
    parsed into dictionaries, the data can subsequently be wrapped and processed using this class.

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
            The datatype names are extracted from the __name__ property of the keys, so the function should be able to
            recognize more or less any type of keys. That said, support beyond the standard key datatypes listed in
            valid_datatypes is not guaranteed.

    Args:
        seed_dictionary: The 'seed' dictionary object to be used by the class. If not provided, the class will generate
            an empty shallow dictionary and use that as the initial object. This argument allows to re-initialize nested
            dictionaries when they are loaded from .yaml or .json files, by passing loaded dictionaries as seeds.
        path_delimiter: The delimiter used to separate keys in string variable paths. It is generally advised
            to stick to the default delimiter for most use cases. Only use custom delimiter if any of the dictionary or
            sub-dictionary keys reserve default delimiter for other purposes (for example, if the delimiter is part of a
            string key). Note, all functions in the class refer to this variable during runtime, so all inputs to the
            class have to use the class delimiter where necessary to avoid unexpected behavior.

    Raises:
        TypeError: If input arguments are not of the supported type.
    """

    def __init__(
        self, seed_dictionary: Optional[dict] = None, path_delimiter: str = "."
    ) -> None:
        # Stores supported key datatypes
        self._valid_datatypes: tuple = ("int", "str", "float", "bool", "NoneType")

        # Verifies input variable types
        if not isinstance(seed_dictionary, (dict, NoneType)):
            message = (
                f"A dictionary or None 'nested_dict' expected when initializing NestedDictionary class instance, but "
                f"encountered '{type(seed_dictionary).__name__}' instead."
            )
            console.error(message=message, error=TypeError)
            raise TypeError(message)  # Fallback, should not be reachable

        elif not isinstance(path_delimiter, str):
            message = (
                f"A string 'path_delimiter' expected when initializing NestedDictionary class instance, but "
                f"encountered '{type(path_delimiter).__name__}' instead."
            )
            console.error(message=message, error=TypeError)
            raise TypeError(message)  # Fallback, should not be reachable

        # Sets class attributes:

        # Initial dictionary object
        if seed_dictionary is not None:
            self._nested_dictionary = (
                seed_dictionary  # If it is provided, uses seed dictionary
            )
        else:
            self._nested_dictionary = (
                dict()
            )  # Creates an empty dictionary as the initial object

        # String path delimiter
        self._path_delimiter = path_delimiter

        # Sets key_datatype variable to a set that stores all key datatypes. This variable is then used by other
        # functions to support the use of string variable paths (where allowed).
        self._key_datatypes = self.extract_key_datatypes()

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
        id_string = (
            f"NestedDictionary(key_datatypes={self._key_datatypes}, path_delimiter='{self._path_delimiter}', "
            f"data={self._nested_dictionary})"
        )
        return id_string

    def extract_key_datatypes(self) -> set[str]:
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
        path_keys: tuple[tuple[Any, ...], ...] = self.extract_nested_variable_paths(
            return_raw=True
        )

        # Initializes an empty set to store unique key datatypes
        unique_types = set()

        # Loops over all key lists
        for keys in path_keys:
            # Updates the set with the name of the types found in the current key tuple (path)
            unique_types.update(type(key).__name__ for key in keys)

        # Returns extracted key datatype names to caller
        return unique_types

    def _convert_key_to_datatype(
        self, key: Any, datatype: Literal["int", "str", "float", "bool", "NoneType"]
    ) -> int | str | float | bool | None:
        """Converts the input key to the requested datatype.

        This method is designed to be used by other class methods when working with dictionary paths and should not
        be called directly by the user.

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
        datatypes = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "NoneType": None,
        }
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
                    f"datatype. Select one of the supported datatypes: {self._valid_datatypes}."
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
            if local_key_dtypes != self._key_datatypes:
                self._key_datatypes = local_key_dtypes

            # For string variable paths, converts the input path keys (formatted as string) into the datatype used by
            # the dictionary keys.
            if isinstance(variable_path, str):
                # If the input argument is a string, ensures it does not end with delimiter
                if variable_path.endswith(self._path_delimiter):
                    custom_error_message = (
                        f"A delimiter-ending variable_path string '{variable_path}' encountered when "
                        f"{operation_description}, which is not allowed. Make sure the variable path ends with a "
                        f"valid key."
                    )
                    raise ValueError(custom_error_message)
                # Additionally, ensures that a string path is accompanied by a valid terminal delimiter value, works
                # equally well for None and any unsupported string options
                elif len(self._key_datatypes) > 1:
                    custom_error_message = (
                        f"A string variable_path '{variable_path}' encountered when {operation_description}, but the "
                        f"dictionary contains mixed key datatypes and does not support string variable "
                        f"path format. Provide a tuple, list or numpy array of keys with each key using one of the "
                        f"supported datatypes ({self._valid_datatypes})."
                    )
                    raise ValueError(custom_error_message)

                # Splits the string path into keys using clas delimiter
                string_keys = variable_path.split(self._path_delimiter)

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
                    keys = [
                        self.convert_key_to_datatype(key=key, datatype=target_dtype)
                        for key in string_keys
                    ]
                except ValueError as e:
                    custom_error_message = (
                        f"Unable to assign the datatypes to keys extracted from variable_path string '{variable_path}' "
                        f"when {operation_description}. Make sure the input path string contains valid "
                        f"keys that can be converted to the datatype '{target_dtype}' used by dictionary keys."
                    )
                    augment_exception_message(
                        e=e, additional_message=custom_error_message
                    )
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
                            f"supported datatypes ({self._valid_datatypes})."
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
                custom_error_message = f"Unable to convert input variable_path '{variable_path}' to keys, when {operation_description}."
                augment_exception_message(e=e, additional_message=custom_error_message)
            raise

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

        Args:
            return_raw: Determines whether the method formats the result as the tuple of key tuples or the tuple of
                delimiter-delimited strings. See notes above for more information.

        Returns:
            If return_raw is true, a tuple of tuples, where each sub-tuple stores a sequence of dictionary path keys.
            If return_raw is false, a tuple of delimiter-delimited path strings.
        """

        def _inner_extract(
            input_dict: dict,
            current_path: Optional[list[Any]] = None,
            *,
            make_raw: bool = False,
        ) -> list[tuple[Any, ...]] | list[str]:
            """Performs the recursive path extraction procedure.

            This function is used to hide recursion variables from end-users, so that they cannot accidentally set them
            to non-default values.

            Most of the extract_nested_variable_paths() method logic comes from this inner function.

            Args:
                input_dict: The dictionary to crawl through. During recursive calls, this variable is used to evaluate
                    sub-dictionaries discovered when crawling the original input dictionary, until the function reaches
                    a non-dictionary value.
                current_path: The ordered list of keys, relative to the top level of the evaluated dictionary. This is
                    used to iteratively construct the sequential key path to each non-dictionary variable. Specifically,
                    recursive function calls add newly discovered keys to the end of the already constructed path key
                    list, preserving the nested hierarchy. This variable is reserved for recursive use, do not change
                    its value!
                make_raw: An alias for the parent function return_raw parameter. This is automatically set to match the
                    parent method return_raw parameter.

            Returns:
                A list of key tuples if return_raw (make_raw) is True and a list of strings otherwise.
            """

            # If current_path is None, creates a new list object to hold the keys. Note, this cannot be a set, as keys
            # at different dictionary levels do not have to be unique relative ot each-other. Therefore, a set may
            # encounter and remove one of the valid duplicated keys along the path. This list is used during recursive
            # calls to keep track of paths being built
            if current_path is None:
                current_path = []

            # This is the overall returned list that keeps track of ALL discovered paths
            paths = []

            # Loops over each key and value extracted from the current view (level) of the nested dictionary
            for key, value in input_dict.items():
                # Appends the local level key to the path tracker list
                new_path = current_path + [key]

                # If the key points to a dictionary, recursively calls the extract function. Passes the current
                # path tracker and the dictionary view returned by evaluated key, to the new function call.
                if isinstance(value, dict):

                    # The recursion keeps winding until it encounters a non-dictionary variable. Once it does, it
                    # causes the stack to unwind until another dictionary is found via the 'for' loop to start
                    # stack winding. As such, the stack will at most use the same number of functions as the number
                    # of nesting levels in the dictionary, which is unlikely to be critically large.
                    # Note, the 'extend' has to be used here over 'append' to iteratively 'stack' node keys as the
                    # function searches for the terminal variable.
                    paths.extend(
                        _inner_extract(
                            input_dict=value, make_raw=make_raw, current_path=new_path
                        )
                    )
                else:
                    # If the key references a non-dictionary variable, formats the constructed key sequence as a tuple
                    # or as a delimited string and appends it to the path list, prior to returning it to caller.
                    # The append operation ensures the path is kept as a separate list object within the final output
                    # list.
                    paths.append(
                        tuple(new_path)
                        if make_raw
                        else self._path_delimiter.join(map(str, new_path))
                    )
            return paths

        # Generates a list of variable paths and converts it to tuple before returning it to the caller. Each path is
        # formatted according to the requested output type by the inner function.
        return tuple(_inner_extract(input_dict=self._nested_dictionary, make_raw=return_raw))

    def read_nested_value(
        self,
        variable_path: str | tuple | list | np.ndarray,
        replace_none_equivalents: bool = False,
        multiprocessing: bool = False,
    ) -> (
        str
        | int
        | float
        | list
        | tuple
        | dict
        | bool
        | None
        | AtaraxisError
        | ValueConverter
        | Any
    ):
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
                variable_path=variable_path,
                operation_description="reading nested value from dictionary",
            )
            handled = False

            # Sets the dictionary view to the highest hierarchy (dictionary itself)
            current_dict_view = self._nested_dictionary

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
                    available_keys_and_types = [
                        [k, type(k).__name__] for k in current_dict_view.keys()
                    ]

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
            if (
                isinstance(current_dict_view, str)
                and current_dict_view in none_values
                and replace_none_equivalents
            ):
                return None
            else:
                # Otherwise, returns the extracted value
                return current_dict_view

        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = f"Unexpected error when reading nested value from dictionary using path '{variable_path}'."
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
                variable_path=variable_path,
                operation_description="writing nested value to dictionary",
            )
            handled = False

            # Generates a copy of the class dictionary as the algorithm uses modification via reference. This way the
            # original dictionary is protected from modification while this function runs. Depending on the function
            # arguments, the original dictionary may still be overwritten with the modified dictionary at the end of the
            # function
            altered_dict = copy.deepcopy(self._nested_dictionary)
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
                self._nested_dictionary = altered_dict
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self._key_datatypes = self.extract_key_datatypes()
            # Otherwise, constructs a new NestedDictionary instance around the altered dictionary and returns this to
            # caller
            else:
                return NestedDictionary(
                    seed_dictionary=altered_dict, path_delimiter=self._path_delimiter
                )
        except Exception as e:
            if not handled:
                # Provides a custom error message
                custom_error_message = f"Unexpected error when writing nested value to dictionary using path '{variable_path}'."
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
                final_key = remaining_keys.pop(
                    0
                )  # Extracts the key from list to variable

                # If the key is found inside the dictionary, removes the variable associated with the key
                if final_key in traversed_dict:
                    del traversed_dict[final_key]

                # If the final key is not found in the dictionary, handles the situation according to whether
                # missing keys are allowed or not. If missing keys are not allowed, issues KeyError
                elif not missing_ok:
                    # Generates a list of lists with each inner list storing the value and datatype for each key in
                    # current dictionary view
                    available_keys_and_types = [
                        [k, type(k).__name__] for k in traversed_dict.keys()
                    ]
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
                    available_keys_and_types = [
                        [k, type(k).__name__] for k in traversed_dict.keys()
                    ]
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
                variable_path=variable_path,
                operation_description="deleting nested value from dictionary",
            )
            handled = False

            # Generates a local copy of the dictionary
            processed_dict = copy.deepcopy(self._nested_dictionary)

            # Initiates recursive processing by calling the first instance of the inner function. Note, the function
            # modifies the dictionary by reference, hence no explicit return statement
            _inner_delete(
                traversed_dict=processed_dict,
                remaining_keys=list(
                    keys
                ),  # Lists are actually more efficient here as they allow in-place modification
                whole_path=variable_path,
                missing_ok=allow_missing,
                delete_empty_directories=delete_empty_sections,
            )

            # If class dictionary modification is preferred, replaces the bundled class dictionary with the processed
            # dictionary
            if modify_class_dictionary:
                self._nested_dictionary = processed_dict
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self._key_datatypes = self.extract_key_datatypes()
            # Otherwise, constructs a new NestedDictionary instance around the processed dictionary and returns this to
            # caller
            else:
                return NestedDictionary(
                    seed_dictionary=processed_dict, path_delimiter=self._path_delimiter
                )

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
                custom_error_message = f"Unexpected error when deleting nested value from dictionary using '{variable_path}' path."
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
        search_mode: Literal[
            "terminal_only", "intermediate_only", "all"
        ] = "terminal_only",
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
            var_paths = self.extract_nested_variable_paths(
                return_raw=True, multiprocessing=False
            )

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
                            storage_list.append(
                                path
                            )  # Preserves order of key discovery

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
                            path_keys = path[
                                : num + 1
                            ]  # Since slicing is uses exclusive end index, uses num+1

                            # Adds unique paths to the storage list. This ensures that the path is not repeatedly
                            # added for multiple paths originating from the same section (when section key matches the
                            # target key).
                            if path_keys not in passed_paths:
                                passed_paths.add(path_keys)
                                storage_list.append(
                                    path_keys
                                )  # Preserves order of key discovery
                # If the function is configured to evaluate intermediate (section) keys only, uses a logic similar to
                # above, but indexes the terminal key out of each evaluated tuple to exclude it from search
                elif search_mode == "intermediate_only":
                    for num, key in enumerate(path[:-1]):
                        if key == target_key:
                            path_keys = path[: num + 1]
                            if path_keys not in passed_paths:
                                passed_paths.add(path_keys)
                                storage_list.append(
                                    path_keys
                                )  # Preserves order of key discovery

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
                    passed_paths = [
                        self._path_delimiter.join(map(str, path))
                        for path in storage_list
                    ]
                    if len(passed_paths) > 1:  # For many paths, returns tuple of tuples
                        return tuple(passed_paths)
                    else:  # For a single path, returns the path as a tuple or string
                        return passed_paths.pop(0)

            # Otherwise, returns None to indicate that no matching paths were found
            else:
                return None

        except Exception as e:
            if not handled:
                custom_error_message = f"Unexpected error when finding the path to the target nested dictionary variable."
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
        valid_datatypes = (
            "str",
            "int",
        )  # Stores allowed datatype options, mostly for error messaging
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
            all_paths = self.extract_nested_variable_paths(
                return_raw=True, multiprocessing=False
            )

            # Converts all keys in all paths to the requested datatype
            try:
                # noinspection PyTypeChecker
                converted_paths = (
                    tuple(
                        self.convert_key_to_datatype(key=key, datatype=datatype)
                        for key in path
                    )
                    for path in all_paths
                )
                converted_paths = tuple(
                    converted_paths
                )  # Converts the outer iterable into a tuple of tuples

            except Exception as e:
                custom_error_message = (
                    f"Unable to convert dictionary keys to '{datatype}' datatype when converting the nested dictionary "
                    f"keys to use a specific datatype."
                )
                augment_exception_message(e=e, additional_message=custom_error_message)
                handled = True
                raise

            # Initializes a new nested dictionary class instance using parent class delimiter and an empty dictionary
            converted_dict = NestedDictionary(
                seed_dictionary={}, path_delimiter=self._path_delimiter
            )

            # Loops over each converted path, retrieves the value associated with original (pre-conversion) path and
            # writes it to the newly created dictionary using the converted path
            try:
                for num, path in enumerate(converted_paths):
                    # Retrieves the value using unconverted path. Note, ensures None-equivalents are NOT converted
                    value = self.read_nested_value(
                        variable_path=all_paths[num],
                        replace_none_equivalents=False,
                        multiprocessing=False,
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
                self._nested_dictionary = copy.deepcopy(
                    converted_dict._nested_dictionary
                )
                # Updates dictionary key datatype tracker in case altered dictionary changed the number of unique
                # datatypes
                self._key_datatypes = self.extract_key_datatypes()
            # Otherwise, returns the newly constructed NestedDictionary instance
            else:
                return converted_dict

        except Exception as e:
            if not handled:
                custom_error_message = f"Unexpected error when converting the nested dictionary keys to use a specific datatype."
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
