from typing import Any, Literal

from _typeshed import Incomplete
from numpy.typing import NDArray

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

    _valid_datatypes: Incomplete
    _nested_dictionary: Incomplete
    _path_delimiter: Incomplete
    _key_datatypes: Incomplete
    def __init__(self, seed_dictionary: dict[Any, Any] | None = None, path_delimiter: str = ".") -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
    @property
    def key_datatypes(self) -> tuple[str, ...]:
        """Returns unique datatypes used by dictionary keys as a sorted tuple."""
    @property
    def path_delimiter(self) -> str:
        """Returns the delimiter used to separate keys in string variable paths."""
    def set_path_delimiter(self, new_delimiter: str) -> None:
        """Sets the path_delimiter class attribute to the provided delimiter value.

        This method can be used to replace the string-path delimiter of an initialized NestedDictionary class.

        Args:
            new_delimiter: The new delimiter to be used for separating keys in string variable paths.

        Raises:
            TypeError: If new_delimiter argument is not a string.

        """
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
    def _convert_key_to_datatype(
        self, key: Any, datatype: Literal["int", "str", "float", "None"]
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
    def extract_nested_variable_paths(self, *, return_raw: bool = False) -> tuple[str] | tuple[tuple[Any, ...], ...]:
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
    def write_nested_value(
        self,
        variable_path: str | tuple[Any, ...] | list[Any] | NDArray[Any],
        value: Any,
        *,
        modify_class_dictionary: bool = True,
        allow_terminal_overwrite: bool = True,
        allow_intermediate_overwrite: bool = False,
    ) -> NestedDictionary | None:
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
    def delete_nested_value(
        self,
        variable_path: str | tuple[Any, ...] | list[Any] | NDArray[Any],
        *,
        modify_class_dictionary: bool = True,
        delete_empty_sections: bool = True,
        allow_missing: bool = False,
    ) -> NestedDictionary | None:
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
    def convert_all_keys_to_datatype(
        self, datatype: Literal["str", "int"], *, modify_class_dictionary: bool = True
    ) -> NestedDictionary | None:
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
