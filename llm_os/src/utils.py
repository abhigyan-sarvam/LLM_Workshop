import logging
import json
from typing import Any, Dict, Union, get_args, get_origin, Optional, List
from pathlib import Path
from importlib import metadata
from rich.logging import RichHandler
from datetime import datetime, timezone, date
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core.core_schema import FieldValidationInfo
from time import perf_counter

PHI_CLI_DIR: Path = Path.home().resolve().joinpath(".phi")
LOGGER_NAME = "phi"

class PhiCliSettings(BaseSettings):
    app_name: str = "phi"

    tmp_token_path: Path = PHI_CLI_DIR.joinpath("tmp_token")
    config_file_path: Path = PHI_CLI_DIR.joinpath("config.json")
    credentials_path: Path = PHI_CLI_DIR.joinpath("credentials.json")
    ai_conversations_path: Path = PHI_CLI_DIR.joinpath("ai_conversations.json")
    auth_token_cookie: str = "__phi_session"
    auth_token_header: str = "X-PHIDATA-AUTH-TOKEN"

    api_runtime: str = "prd"
    api_enabled: bool = True
    api_url: str = Field("https://api.phidata.com", validate_default=True)
    signin_url: str = Field("https://phidata.app/login", validate_default=True)

    model_config = SettingsConfigDict(env_prefix="PHI_")

    @field_validator("api_runtime", mode="before")
    def validate_runtime_env(cls, v):
        """Validate api_runtime."""

        valid_api_runtimes = ["dev", "stg", "prd"]
        if v not in valid_api_runtimes:
            raise ValueError(f"Invalid api_runtime: {v}")

        return v

    @field_validator("signin_url", mode="before")
    def update_signin_url(cls, v, info: FieldValidationInfo):
        api_runtime = info.data["api_runtime"]
        if api_runtime == "dev":
            return "http://localhost:3000/login"
        elif api_runtime == "stg":
            return "https://stgphi.com/login"
        else:
            return "https://phidata.app/login"

    @field_validator("api_url", mode="before")
    def update_api_url(cls, v, info: FieldValidationInfo):
        api_runtime = info.data["api_runtime"]
        if api_runtime == "dev":
            from os import getenv

            if getenv("PHI_RUNTIME") == "docker":
                return "http://host.docker.internal:7070"
            return "http://localhost:7070"
        elif api_runtime == "stg":
            return "https://api.stgphi.com"
        else:
            return "https://api.phidata.com"

phi_cli_settings = PhiCliSettings()

def get_logger(logger_name: str) -> logging.Logger:
    # https://rich.readthedocs.io/en/latest/reference/logging.html#rich.logging.RichHandler
    # https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
    rich_handler = RichHandler(
        show_time=False,
        rich_tracebacks=False,
        show_path=True if phi_cli_settings.api_runtime == "dev" else False,
        tracebacks_show_locals=False,
    )
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    _logger = logging.getLogger(logger_name)
    _logger.addHandler(rich_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    return _logger

logger: logging.Logger = get_logger(LOGGER_NAME)


def set_log_level_to_debug():
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)

def current_datetime() -> datetime:
    return datetime.now()

def current_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)

def current_datetime_utc_str() -> str:
    return current_datetime_utc().strftime("%Y-%m-%dT%H:%M:%S")

def get_json_type_for_py_type(arg: str) -> str:
    """
    Get the JSON schema type for a given type.
    :param arg: The type to get the JSON schema type for.
    :return: The JSON schema type.

    See: https://json-schema.org/understanding-json-schema/reference/type.html#type-specific-keywords
    """
    # logger.info(f"Getting JSON type for: {arg}")
    if arg in ("int", "float"):
        return "number"
    elif arg == "str":
        return "string"
    elif arg == "bool":
        return "boolean"
    elif arg in ("NoneType", "None"):
        return "null"
    return arg

def get_json_schema_for_arg(t: Any) -> Optional[Any]:
    # logger.info(f"Getting JSON schema for arg: {t}")
    json_schema = None
    type_args = get_args(t)
    # logger.info(f"Type args: {type_args}")
    type_origin = get_origin(t)
    # logger.info(f"Type origin: {type_origin}")
    if type_origin is not None:
        if type_origin is list:
            json_schema_for_items = get_json_schema_for_arg(type_args[0])
            json_schema = {"type": "array", "items": json_schema_for_items}
        elif type_origin is dict:
            json_schema = {"type": "object", "properties": {}}
        elif type_origin is Union:
            json_schema = {"type": [get_json_type_for_py_type(arg.__name__) for arg in type_args]}
    else:
        json_schema = {"type": get_json_type_for_py_type(t.__name__)}
    return json_schema

def get_json_schema(type_hints: Dict[str, Any]) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    for k, v in type_hints.items():
        # logger.info(f"Parsing arg: {k} | {v}")
        if k == "return":
            continue
        arg_json_schema = get_json_schema_for_arg(v)
        if arg_json_schema is not None:
            # logger.info(f"json_schema: {arg_json_schema}")
            json_schema["properties"][k] = arg_json_schema
        else:
            logger.warning(f"Could not parse argument {k} of type {v}")
    return json_schema

def merge_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    """
    Recursively merges two dictionaries.
    If there are conflicting keys, values from 'b' will take precedence.

    @params:
    a (Dict[str, Any]): The first dictionary to be merged.
    b (Dict[str, Any]): The second dictionary, whose values will take precedence.

    Returns:
    None: The function modifies the first dictionary in place.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dictionaries(a[key], b[key])
        else:
            a[key] = b[key]

def get_text_from_message(message: Union[List, Dict, str]) -> str:
    """Return the user texts from the message"""

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_messages = []
        if len(message) == 0:
            return ""

        if "type" in message[0]:
            for m in message:
                m_type = m.get("type")
                if m_type is not None and isinstance(m_type, str):
                    m_value = m.get(m_type)
                    if m_value is not None and isinstance(m_value, str):
                        if m_type == "text":
                            text_messages.append(m_value)
                        # if m_type == "image_url":
                        #     text_messages.append(f"Image: {m_value}")
                        # else:
                        #     text_messages.append(f"{m_type}: {m_value}")
        elif "role" in message[0]:
            for m in message:
                m_role = m.get("role")
                if m_role is not None and isinstance(m_role, str):
                    m_content = m.get("content")
                    if m_content is not None and isinstance(m_content, str):
                        if m_role == "user":
                            text_messages.append(m_content)
        if len(text_messages) > 0:
            return "\n".join(text_messages)
    return ""

from src.tools import Function, FunctionCall
def get_function_call(
    name: str,
    arguments: Optional[str] = None,
    call_id: Optional[str] = None,
    functions: Optional[Dict[str, Function]] = None,
) -> Optional[FunctionCall]:
    logger.debug(f"Getting function {name}")
    if functions is None:
        return None

    function_to_call: Optional[Function] = None
    if name in functions:
        function_to_call = functions[name]
    if function_to_call is None:
        logger.error(f"Function {name} not found")
        return None

    function_call = FunctionCall(function=function_to_call)
    if call_id is not None:
        function_call.call_id = call_id
    if arguments is not None and arguments != "":
        try:
            if function_to_call.sanitize_arguments:
                if "None" in arguments:
                    arguments = arguments.replace("None", "null")
                if "True" in arguments:
                    arguments = arguments.replace("True", "true")
                if "False" in arguments:
                    arguments = arguments.replace("False", "false")
            _arguments = json.loads(arguments)
        except Exception as e:
            logger.error(f"Unable to decode function arguments:\n{arguments}\nError: {e}")
            function_call.error = (
                f"Error while decoding function arguments: {e}\n\n"
                f"Please make sure we can json.loads() the arguments and retry."
            )
            return function_call

        if not isinstance(_arguments, dict):
            logger.error(f"Function arguments are not a valid JSON object: {arguments}")
            function_call.error = "Function arguments are not a valid JSON object.\n\n Please fix and retry."
            return function_call

        try:
            clean_arguments: Dict[str, Any] = {}
            for k, v in _arguments.items():
                if isinstance(v, str):
                    _v = v.strip().lower()
                    if _v in ("none", "null"):
                        clean_arguments[k] = None
                    elif _v == "true":
                        clean_arguments[k] = True
                    elif _v == "false":
                        clean_arguments[k] = False
                    else:
                        clean_arguments[k] = v.strip()
                else:
                    clean_arguments[k] = v

            function_call.arguments = clean_arguments
        except Exception as e:
            logger.error(f"Unable to parsing function arguments:\n{arguments}\nError: {e}")
            function_call.error = f"Error while parsing function arguments: {e}\n\n Please fix and retry."
            return function_call
    return function_call

def get_function_call_for_tool_call(
    tool_call: Dict[str, Any], functions: Optional[Dict[str, Function]] = None
) -> Optional[FunctionCall]:
    if tool_call.get("type") == "function":
        _tool_call_id = tool_call.get("id")
        _tool_call_function = tool_call.get("function")
        if _tool_call_function is not None:
            _tool_call_function_name = _tool_call_function.get("name")
            _tool_call_function_arguments_str = _tool_call_function.get("arguments")
            if _tool_call_function_name is not None:
                return get_function_call(
                    name=_tool_call_function_name,
                    arguments=_tool_call_function_arguments_str,
                    call_id=_tool_call_id,
                    functions=functions,
                )
    return None

def extract_tool_call_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extracting the content between the tags
    return text[start_index:end_index].strip()

def remove_tool_calls_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    """Remove multiple tool calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text

def extract_tool_from_xml(xml_str):
    # Find tool_name
    tool_name_start = xml_str.find("<tool_name>") + len("<tool_name>")
    tool_name_end = xml_str.find("</tool_name>")
    tool_name = xml_str[tool_name_start:tool_name_end].strip()

    # Find and process parameters block
    params_start = xml_str.find("<parameters>") + len("<parameters>")
    params_end = xml_str.find("</parameters>")
    parameters_block = xml_str[params_start:params_end].strip()

    # Extract individual parameters
    arguments = {}
    while parameters_block:
        # Find the next tag and its closing
        tag_start = parameters_block.find("<") + 1
        tag_end = parameters_block.find(">")
        tag_name = parameters_block[tag_start:tag_end]

        # Find the tag's closing counterpart
        value_start = tag_end + 1
        value_end = parameters_block.find(f"</{tag_name}>")
        value = parameters_block[value_start:value_end].strip()

        # Add to arguments
        arguments[tag_name] = value

        # Move past this tag
        parameters_block = parameters_block[value_end + len(f"</{tag_name}>") :].strip()

    return {"tool_name": tool_name, "parameters": arguments}

def remove_function_calls_from_string(
    text: str, start_tag: str = "<function_calls>", end_tag: str = "</function_calls>"
):
    """Remove multiple function calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text

def remove_indent(s: Optional[str]) -> Optional[str]:
    """
    Remove the indent from a string.

    Args:
        s (str): String to remove indent from

    Returns:
        str: String with indent removed
    """
    if s is not None and isinstance(s, str):
        return "\n".join([line.strip() for line in s.split("\n")])
    return None

def read_yaml_file(file_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if file_path is not None and file_path.exists() and file_path.is_file():
        import yaml

        logger.debug(f"Reading {file_path}")
        data_from_file = yaml.safe_load(file_path.read_text())
        if data_from_file is not None and isinstance(data_from_file, dict):
            return data_from_file
        else:
            logger.error(f"Invalid file: {file_path}")
    return None

def write_yaml_file(file_path: Optional[Path], data: Optional[Dict[str, Any]], **kwargs) -> None:
    if file_path is not None and data is not None:
        import yaml

        logger.debug(f"Writing {file_path}")
        file_path.write_text(yaml.safe_dump(data, **kwargs))

class Timer:
    """Timer class for timing code execution"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return self.elapsed_time or (perf_counter() - self.start_time) if self.start_time else 0.0

    def start(self) -> float:
        self.start_time = perf_counter()
        return self.start_time

    def stop(self) -> float:
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
        return self.end_time

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime) or isinstance(o, date):
            return o.isoformat()

        if isinstance(o, Path):
            return str(o)

        return json.JSONEncoder.default(self, o)

def read_json_file(file_path: Optional[Path]) -> Optional[Union[Dict, List]]:
    if file_path is not None and file_path.exists() and file_path.is_file():
        logger.debug(f"Reading {file_path}")
        return json.loads(file_path.read_text())
    return None

def write_json_file(file_path: Optional[Path], data: Optional[Union[Dict, List]], **kwargs) -> None:
    if file_path is not None and data is not None:
        logger.debug(f"Writing {file_path}")
        file_path.write_text(json.dumps(data, cls=CustomJSONEncoder, indent=4, **kwargs))
