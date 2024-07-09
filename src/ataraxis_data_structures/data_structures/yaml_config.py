from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import yaml as yaml
from ..standalone_methods.data_manipulation_methods import ensure_list


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
