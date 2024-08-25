from typing import Any, Dict, Optional, Callable, List, get_type_hints
from pydantic import BaseModel, validate_call
from collections import OrderedDict
import json
import math
from os import getenv
from pathlib import Path
import runpy
import functools
from src.utils import logger, get_json_schema

try:
    from exa_py import Exa
except ImportError:
    raise ImportError("`exa_py` not installed. Please install using `pip install exa_py`")
try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError("`duckduckgo-search` not installed. Please install using `pip install duckduckgo-search`")


class Function(BaseModel):
    """Model for Functions"""

    # The name of the function to be called.
    # Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: str
    # A description of what the function does, used by the model to choose when and how to call the function.
    description: Optional[str] = None
    # The parameters the functions accepts, described as a JSON Schema object.
    # To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    entrypoint: Optional[Callable] = None

    # If True, the arguments are sanitized before being passed to the function.
    sanitize_arguments: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={"name", "description", "parameters"})

    @classmethod
    def from_callable(cls, c: Callable) -> "Function":
        from inspect import getdoc

        parameters = {"type": "object", "properties": {}}
        try:
            # logger.info(f"Getting type hints for {c}")
            type_hints = get_type_hints(c)
            # logger.info(f"Type hints for {c}: {type_hints}")
            # logger.info(f"Getting JSON schema for {type_hints}")
            parameters = get_json_schema(type_hints)
            # logger.info(f"JSON schema for {c}: {parameters}")
            # logger.debug(f"Type hints for {c.__name__}: {type_hints}")
        except Exception as e:
            logger.warning(f"Could not parse args for {c.__name__}: {e}")

        return cls(
            name=c.__name__,
            description=getdoc(c),
            parameters=parameters,
            entrypoint=validate_call(c),
        )

    def get_type_name(self, t):
        name = str(t)
        if "list" in name or "dict" in name:
            return name
        else:
            return t.__name__

    def get_definition_for_prompt(self) -> Optional[str]:
        """Returns a function definition that can be used in a prompt."""
        import json

        if self.entrypoint is None:
            return None

        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)

        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return json.dumps(function_info, indent=2)

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        """Returns a function definition that can be used in a prompt."""

        if self.entrypoint is None:
            return None

        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)

        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return function_info

class FunctionCall(BaseModel):
    """Model for Function Calls"""

    # The function to be called.
    function: Function
    # The arguments to call the function with.
    arguments: Optional[Dict[str, Any]] = None
    # The result of the function call.
    result: Optional[Any] = None
    # The ID of the function call.
    call_id: Optional[str] = None

    # Error while parsing arguments or running the function.
    error: Optional[str] = None

    def get_call_str(self) -> str:
        """Returns a string representation of the function call."""
        if self.arguments is None:
            return f"{self.function.name}()"

        trimmed_arguments = {}
        for k, v in self.arguments.items():
            if isinstance(v, str) and len(v) > 100:
                trimmed_arguments[k] = "..."
            else:
                trimmed_arguments[k] = v
        call_str = f"{self.function.name}({', '.join([f'{k}={v}' for k, v in trimmed_arguments.items()])})"
        return call_str

    def execute(self) -> bool:
        """Runs the function call.

        @return: True if the function call was successful, False otherwise.
        """
        if self.function.entrypoint is None:
            return False

        logger.debug(f"Running: {self.get_call_str()}")

        # Call the function with no arguments if none are provided.
        if self.arguments is None:
            try:
                self.result = self.function.entrypoint()
                return True
            except Exception as e:
                logger.warning(f"Could not run function {self.get_call_str()}")
                logger.exception(e)
                self.error = str(e)
                return False

        try:
            self.result = self.function.entrypoint(**self.arguments)
            return True
        except Exception as e:
            logger.warning(f"Could not run function {self.get_call_str()}")
            logger.exception(e)
            self.error = str(e)
            return False

class Toolkit:
    def __init__(self, name: str = "toolkit"):
        self.name: str = name
        self.functions: Dict[str, Function] = OrderedDict()

    def register(self, function: Callable, sanitize_arguments: bool = True):
        try:
            f = Function.from_callable(function)
            f.sanitize_arguments = sanitize_arguments
            self.functions[f.name] = f
            logger.debug(f"Function: {f.name} registered with {self.name}")
            # logger.debug(f"Json Schema: {f.to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to create Function for: {function.__name__}")
            raise e

    def instructions(self) -> str:
        return ""

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>"

    def __str__(self):
        return self.__repr__()

class Tool(BaseModel):
    """Model for Tools"""

    # The type of tool
    type: str
    # The function to be called if type = "function"
    function: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

class ExaTools(Toolkit):
    def __init__(
        self,
        text: bool = True,
        text_length_limit: int = 1000,
        highlights: bool = True,
        api_key: Optional[str] = None,
        num_results: Optional[int] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
        type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        show_results: bool = False,
    ):
        super().__init__(name="exa")

        self.api_key = api_key or getenv("EXA_API_KEY")
        if not self.api_key:
            logger.error("EXA_API_KEY not set. Please set the EXA_API_KEY environment variable.")

        self.show_results = show_results

        self.text: bool = text
        self.text_length_limit: int = text_length_limit
        self.highlights: bool = highlights
        self.num_results: Optional[int] = num_results
        self.start_crawl_date: Optional[str] = start_crawl_date
        self.end_crawl_date: Optional[str] = end_crawl_date
        self.start_published_date: Optional[str] = start_published_date
        self.end_published_date: Optional[str] = end_published_date
        self.use_autoprompt: Optional[bool] = use_autoprompt
        self.type: Optional[str] = type
        self.include_domains: Optional[List[str]] = include_domains
        self.category: Optional[str] = category

        self.register(self.search_exa)

    def search_exa(self, query: str, num_results: int = 5) -> str:
        """Use this function to search Exa (a web search engine) for a query.

        Args:
            query (str): The query to search for.
            num_results (int): Number of results to return. Defaults to 5.

        Returns:
            str: The search results in JSON format.
        """
        if not self.api_key:
            return "Please set the EXA_API_KEY"

        try:
            exa = Exa(self.api_key)
            logger.info(f"Searching exa for: {query}")
            search_kwargs: Dict[str, Any] = {
                "text": self.text,
                "highlights": self.highlights,
                "num_results": self.num_results or num_results,
                "start_crawl_date": self.start_crawl_date,
                "end_crawl_date": self.end_crawl_date,
                "start_published_date": self.start_published_date,
                "end_published_date": self.end_published_date,
                "use_autoprompt": self.use_autoprompt,
                "type": self.type,
                "category": self.category,
                "include_domains": self.include_domains,
            }
            # Clean up the kwargs
            search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}
            exa_results = exa.search_and_contents(query, **search_kwargs)
            exa_results_parsed = []
            for result in exa_results.results:
                result_dict = {"url": result.url}
                if result.title:
                    result_dict["title"] = result.title
                if result.author and result.author != "":
                    result_dict["author"] = result.author
                if result.published_date:
                    result_dict["published_date"] = result.published_date
                if result.text:
                    _text = result.text
                    if self.text_length_limit:
                        _text = _text[: self.text_length_limit]
                    result_dict["text"] = _text
                if self.highlights:
                    try:
                        if result.highlights:  # type: ignore
                            result_dict["highlights"] = result.highlights  # type: ignore
                    except Exception as e:
                        logger.debug(f"Failed to get highlights {e}")
                exa_results_parsed.append(result_dict)
            parsed_results = json.dumps(exa_results_parsed, indent=4)
            if self.show_results:
                logger.info(parsed_results)
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to search exa {e}")
            return f"Error: {e}"

class Calculator(Toolkit):
    def __init__(
        self,
        add: bool = True,
        subtract: bool = True,
        multiply: bool = True,
        divide: bool = True,
        exponentiate: bool = False,
        factorial: bool = False,
        is_prime: bool = False,
        square_root: bool = False,
    ):
        super().__init__(name="calculator")

        # Register functions in the toolkit
        if add:
            self.register(self.add)
        if subtract:
            self.register(self.subtract)
        if multiply:
            self.register(self.multiply)
        if divide:
            self.register(self.divide)
        if exponentiate:
            self.register(self.exponentiate)
        if factorial:
            self.register(self.factorial)
        if is_prime:
            self.register(self.is_prime)
        if square_root:
            self.register(self.square_root)

    def add(self, a: float, b: float) -> str:
        """Add two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a + b
        logger.info(f"Adding {a} and {b} to get {result}")
        return json.dumps({"operation": "addition", "result": result})

    def subtract(self, a: float, b: float) -> str:
        """Subtract second number from first and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a - b
        logger.info(f"Subtracting {b} from {a} to get {result}")
        return json.dumps({"operation": "subtraction", "result": result})

    def multiply(self, a: float, b: float) -> str:
        """Multiply two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a * b
        logger.info(f"Multiplying {a} and {b} to get {result}")
        return json.dumps({"operation": "multiplication", "result": result})

    def divide(self, a: float, b: float) -> str:
        """Divide first number by second and return the result.

        Args:
            a (float): Numerator.
            b (float): Denominator.

        Returns:
            str: JSON string of the result.
        """
        if b == 0:
            logger.error("Attempt to divide by zero")
            return json.dumps({"operation": "division", "error": "Division by zero is undefined"})
        try:
            result = a / b
        except Exception as e:
            return json.dumps({"operation": "division", "error": e, "result": "Error"})
        logger.info(f"Dividing {a} by {b} to get {result}")
        return json.dumps({"operation": "division", "result": result})

    def exponentiate(self, a: float, b: float) -> str:
        """Raise first number to the power of the second number and return the result.

        Args:
            a (float): Base.
            b (float): Exponent.

        Returns:
            str: JSON string of the result.
        """
        result = math.pow(a, b)
        logger.info(f"Raising {a} to the power of {b} to get {result}")
        return json.dumps({"operation": "exponentiation", "result": result})

    def factorial(self, n: int) -> str:
        """Calculate the factorial of a number and return the result.

        Args:
            n (int): Number to calculate the factorial of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            logger.error("Attempt to calculate factorial of a negative number")
            return json.dumps({"operation": "factorial", "error": "Factorial of a negative number is undefined"})
        result = math.factorial(n)
        logger.info(f"Calculating factorial of {n} to get {result}")
        return json.dumps({"operation": "factorial", "result": result})

    def is_prime(self, n: int) -> str:
        """Check if a number is prime and return the result.

        Args:
            n (int): Number to check if prime.

        Returns:
            str: JSON string of the result.
        """
        if n <= 1:
            return json.dumps({"operation": "prime_check", "result": False})
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return json.dumps({"operation": "prime_check", "result": False})
        return json.dumps({"operation": "prime_check", "result": True})

    def square_root(self, n: float) -> str:
        """Calculate the square root of a number and return the result.

        Args:
            n (float): Number to calculate the square root of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            logger.error("Attempt to calculate square root of a negative number")
            return json.dumps({"operation": "square_root", "error": "Square root of a negative number is undefined"})

        result = math.sqrt(n)
        logger.info(f"Calculating square root of {n} to get {result}")
        return json.dumps({"operation": "square_root", "result": result})

class DuckDuckGo(Toolkit):
    def __init__(
        self,
        search: bool = True,
        news: bool = True,
        fixed_max_results: Optional[int] = None,
        headers: Optional[Any] = None,
        proxy: Optional[str] = None,
        proxies: Optional[Any] = None,
        timeout: Optional[int] = 10,
    ):
        super().__init__(name="duckduckgo")

        self.headers: Optional[Any] = headers
        self.proxy: Optional[str] = proxy
        self.proxies: Optional[Any] = proxies
        self.timeout: Optional[int] = timeout
        self.fixed_max_results: Optional[int] = fixed_max_results
        if search:
            self.register(self.duckduckgo_search)
        if news:
            self.register(self.duckduckgo_news)

    def duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search DuckDuckGo for a query.

        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The result from DuckDuckGo.
        """
        logger.debug(f"Searching DDG for: {query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout)
        return json.dumps(ddgs.text(keywords=query, max_results=(self.fixed_max_results or max_results)), indent=2)

    def duckduckgo_news(self, query: str, max_results: int = 5) -> str:
        """Use this function to get the latest news from DuckDuckGo.

        Args:
            query(str): The query to search for.
            max_results (optional, default=5): The maximum number of results to return.

        Returns:
            The latest news from DuckDuckGo.
        """
        logger.debug(f"Searching DDG news for: {query}")
        ddgs = DDGS(headers=self.headers, proxy=self.proxy, proxies=self.proxies, timeout=self.timeout)
        return json.dumps(ddgs.news(keywords=query, max_results=(self.fixed_max_results or max_results)), indent=2)

class FileTools(Toolkit):
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        save_files: bool = True,
        read_files: bool = True,
        list_files: bool = True,
    ):
        super().__init__(name="file_tools")

        self.base_dir: Path = base_dir or Path.cwd()
        if save_files:
            self.register(self.save_file, sanitize_arguments=False)
        if read_files:
            self.register(self.read_file)
        if list_files:
            self.register(self.list_files)

    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        :param contents: The contents to save.
        :param file_name: The name of the file to save to.
        :param overwrite: Overwrite the file if it already exists.
        :return: The file name if successful, otherwise returns an error message.
        """
        try:
            file_path = self.base_dir.joinpath(file_name)
            logger.debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents)
            logger.info(f"Saved: {file_path}")
            return str(file_name)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_file(self, file_name: str) -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.

        :param file_name: The name of the file to read.
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Reading file: {file_name}")
            file_path = self.base_dir.joinpath(file_name)
            contents = file_path.read_text()
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        """Returns a list of files in the base directory

        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Reading files in : {self.base_dir}")
            return json.dumps([str(file_path) for file_path in self.base_dir.iterdir()], indent=4)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

@functools.lru_cache(maxsize=None)
def warn() -> None:
    logger.warning("PythonTools can run arbitrary code, please provide human supervision.")

class PythonTools(Toolkit):
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        save_and_run: bool = True,
        pip_install: bool = False,
        run_code: bool = False,
        list_files: bool = False,
        run_files: bool = False,
        read_files: bool = False,
        safe_globals: Optional[dict] = None,
        safe_locals: Optional[dict] = None,
    ):
        super().__init__(name="python_tools")

        self.base_dir: Path = base_dir or Path.cwd()

        # Restricted global and local scope
        self.safe_globals: dict = safe_globals or globals()
        self.safe_locals: dict = safe_locals or locals()

        if run_code:
            self.register(self.run_python_code, sanitize_arguments=False)
        if save_and_run:
            self.register(self.save_to_file_and_run, sanitize_arguments=False)
        if pip_install:
            self.register(self.pip_install_package)
        if run_files:
            self.register(self.run_python_file_return_variable)
        if read_files:
            self.register(self.read_file)
        if list_files:
            self.register(self.list_files)

    def save_to_file_and_run(
        self, file_name: str, code: str, variable_to_return: Optional[str] = None, overwrite: bool = True
    ) -> str:
        """This function saves Python code to a file called `file_name` and then runs it.
        If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
        If failed, returns an error message.

        Make sure the file_name ends with `.py`

        :param file_name: The name of the file the code will be saved to.
        :param code: The code to save and run.
        :param variable_to_return: The variable to return.
        :param overwrite: Overwrite the file if it already exists.
        :return: if run is successful, the value of `variable_to_return` if provided else file name.
        """
        try:
            warn()
            file_path = self.base_dir.joinpath(file_name)
            logger.debug(f"Saving code to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(code)
            logger.info(f"Saved: {file_path}")
            logger.info(f"Running {file_path}")
            globals_after_run = runpy.run_path(str(file_path), init_globals=self.safe_globals, run_name="__main__")

            if variable_to_return:
                variable_value = globals_after_run.get(variable_to_return)
                if variable_value is None:
                    return f"Variable {variable_to_return} not found"
                logger.debug(f"Variable {variable_to_return} value: {variable_value}")
                return str(variable_value)
            else:
                return f"successfully ran {str(file_path)}"
        except Exception as e:
            logger.error(f"Error saving and running code: {e}")
            return f"Error saving and running code: {e}"

    def run_python_file_return_variable(self, file_name: str, variable_to_return: Optional[str] = None) -> str:
        """This function runs code in a Python file.
        If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
        If failed, returns an error message.

        :param file_name: The name of the file to run.
        :param variable_to_return: The variable to return.
        :return: if run is successful, the value of `variable_to_return` if provided else file name.
        """
        try:
            warn()
            file_path = self.base_dir.joinpath(file_name)

            logger.info(f"Running {file_path}")
            globals_after_run = runpy.run_path(str(file_path), init_globals=self.safe_globals, run_name="__main__")
            if variable_to_return:
                variable_value = globals_after_run.get(variable_to_return)
                if variable_value is None:
                    return f"Variable {variable_to_return} not found"
                logger.debug(f"Variable {variable_to_return} value: {variable_value}")
                return str(variable_value)
            else:
                return f"successfully ran {str(file_path)}"
        except Exception as e:
            logger.error(f"Error running file: {e}")
            return f"Error running file: {e}"

    def read_file(self, file_name: str) -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.

        :param file_name: The name of the file to read.
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Reading file: {file_name}")
            file_path = self.base_dir.joinpath(file_name)
            contents = file_path.read_text()
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        """Returns a list of files in the base directory

        :return: Comma separated list of files in the base directory.
        """
        try:
            logger.info(f"Reading files in : {self.base_dir}")
            files = [str(file_path.name) for file_path in self.base_dir.iterdir()]
            return ", ".join(files)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

    def run_python_code(self, code: str, variable_to_return: Optional[str] = None) -> str:
        """This function to runs Python code in the current environment.
        If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
        If failed, returns an error message.

        Returns the value of `variable_to_return` if successful, otherwise returns an error message.

        :param code: The code to run.
        :param variable_to_return: The variable to return.
        :return: value of `variable_to_return` if successful, otherwise returns an error message.
        """
        try:
            warn()

            logger.debug(f"Running code:\n\n{code}\n\n")
            exec(code, self.safe_globals, self.safe_locals)

            if variable_to_return:
                variable_value = self.safe_locals.get(variable_to_return)
                if variable_value is None:
                    return f"Variable {variable_to_return} not found"
                logger.debug(f"Variable {variable_to_return} value: {variable_value}")
                return str(variable_value)
            else:
                return "successfully ran python code"
        except Exception as e:
            logger.error(f"Error running python code: {e}")
            return f"Error running python code: {e}"

    def pip_install_package(self, package_name: str) -> str:
        """This function installs a package using pip in the current environment.
        If successful, returns a success message.
        If failed, returns an error message.

        :param package_name: The name of the package to install.
        :return: success message if successful, otherwise returns an error message.
        """
        try:
            warn()

            logger.debug(f"Installing package {package_name}")
            import sys
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"successfully installed package {package_name}"
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return f"Error installing package {package_name}: {e}"
