from os import getenv
from typing import Union, Dict, List, Optional, Any
from pathlib import Path
from importlib import metadata
from pydantic import field_validator, Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core.core_schema import FieldValidationInfo
from httpx import Client as HttpxClient, AsyncClient as HttpxAsyncClient, Response
from src.utils import read_json_file, write_json_file, logger
from dataclasses import dataclass

PHI_CLI_DIR: Path = Path.home().resolve().joinpath(".phi")
PHI_API_KEY_ENV_VAR: str = "PHI_API_KEY"
PHI_WS_KEY_ENV_VAR: str = "PHI_WS_KEY"


def save_auth_token(auth_token: str):
    # logger.debug(f"Storing {auth_token} to {str(phi_cli_settings.credentials_path)}")
    _data = {"token": auth_token}
    write_json_file(phi_cli_settings.credentials_path, _data)

def read_auth_token() -> Optional[str]:
    # logger.debug(f"Reading token from {str(phi_cli_settings.credentials_path)}")
    _data: Dict = read_json_file(phi_cli_settings.credentials_path)  # type: ignore
    if _data is None:
        return None

    try:
        return _data.get("token")
    except Exception:
        pass
    return None

def invalid_response(r: Response) -> bool:
    """Returns true if the response is invalid"""

    if r.status_code >= 400:
        return True
    return False


class PhiCliSettings(BaseSettings):
    app_name: str = "phi"
    app_version: str = metadata.version("phidata")

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


class Api:
    def __init__(self):
        self.headers: Dict[str, str] = {
            "user-agent": f"{phi_cli_settings.app_name}/{phi_cli_settings.app_version}",
            "Content-Type": "application/json",
        }
        self._auth_token: Optional[str] = None
        self._authenticated_headers = None

    @property
    def auth_token(self) -> Optional[str]:
        if self._auth_token is None:
            try:
                self._auth_token = read_auth_token()
            except Exception as e:
                logger.debug(f"Failed to read auth token: {e}")
        return self._auth_token

    @property
    def authenticated_headers(self) -> Dict[str, str]:
        if self._authenticated_headers is None:
            self._authenticated_headers = self.headers.copy()
            token = self.auth_token
            if token is not None:
                self._authenticated_headers[phi_cli_settings.auth_token_header] = token
        return self._authenticated_headers

    def Client(self) -> HttpxClient:
        return HttpxClient(
            base_url=phi_cli_settings.api_url,
            headers=self.headers,
            timeout=60,
        )

    def AuthenticatedClient(self) -> HttpxClient:
        return HttpxClient(
            base_url=phi_cli_settings.api_url,
            headers=self.authenticated_headers,
            timeout=60,
        )

    def AsyncClient(self) -> HttpxAsyncClient:
        return HttpxAsyncClient(
            base_url=phi_cli_settings.api_url,
            headers=self.headers,
            timeout=60,
        )

    def AuthenticatedAsyncClient(self) -> HttpxAsyncClient:
        return HttpxAsyncClient(
            base_url=phi_cli_settings.api_url,
            headers=self.authenticated_headers,
            timeout=60,
        )

api = Api()

@dataclass
class ApiRoutes:
    # user paths
    USER_HEALTH: str = "/v1/user/health"
    USER_READ: str = "/v1/user/read"
    USER_CREATE: str = "/v1/user/create"
    USER_UPDATE: str = "/v1/user/update"
    USER_SIGN_IN: str = "/v1/user/signin"
    USER_CLI_AUTH: str = "/v1/user/cliauth"
    USER_AUTHENTICATE: str = "/v1/user/authenticate"
    USER_AUTH_REFRESH: str = "/v1/user/authrefresh"

    # workspace paths
    WORKSPACE_HEALTH: str = "/v1/workspace/health"
    WORKSPACE_CREATE: str = "/v1/workspace/create"
    WORKSPACE_UPDATE: str = "/v1/workspace/update"
    WORKSPACE_DELETE: str = "/v1/workspace/delete"
    WORKSPACE_EVENT_CREATE: str = "/v1/workspace/event/create"
    WORKSPACE_UPDATE_PRIMARY: str = "/v1/workspace/update/primary"
    WORKSPACE_READ_PRIMARY: str = "/v1/workspace/read/primary"
    WORKSPACE_READ_AVAILABLE: str = "/v1/workspace/read/available"

    # assistant paths
    ASSISTANT_RUN_CREATE: str = "/v1/assistant/run/create"
    ASSISTANT_EVENT_CREATE: str = "/v1/assistant/event/create"

    # prompt paths
    PROMPT_REGISTRY_SYNC: str = "/v1/prompt/registry/sync"
    PROMPT_TEMPLATE_SYNC: str = "/v1/prompt/template/sync"

    # ai paths
    AI_CONVERSATION_CREATE: str = "/v1/ai/conversation/create"
    AI_CONVERSATION_CHAT: str = "/v1/ai/conversation/chat"
    AI_CONVERSATION_CHAT_WS: str = "/v1/ai/conversation/chat_ws"

    # llm paths
    OPENAI_CHAT: str = "/v1/llm/openai/chat"
    OPENAI_EMBEDDING: str = "/v1/llm/openai/embedding"

class AssistantRunCreate(BaseModel):
    """Data sent to API to create an assistant run"""

    run_id: str
    assistant_data: Optional[Dict[str, Any]] = None

class AssistantEventCreate(BaseModel):
    """Data sent to API to create a new assistant event"""

    run_id: str
    assistant_data: Optional[Dict[str, Any]] = None
    event_type: str
    event_data: Optional[Dict[str, Any]] = None

def create_assistant_run(run: AssistantRunCreate) -> bool:
    if not phi_cli_settings.api_enabled:
        return True

    logger.debug("--o-o-- Creating Assistant Run")
    with api.AuthenticatedClient() as api_client:
        try:
            r: Response = api_client.post(
                ApiRoutes.ASSISTANT_RUN_CREATE,
                headers={
                    "Authorization": f"Bearer {getenv(PHI_API_KEY_ENV_VAR)}",
                    "PHI-WORKSPACE": f"{getenv(PHI_WS_KEY_ENV_VAR)}",
                },
                json={
                    "run": run.model_dump(exclude_none=True),
                    # "workspace": assistant_workspace.model_dump(exclude_none=True),
                },
            )
            if invalid_response(r):
                return False

            response_json: Union[Dict, List] = r.json()
            if response_json is None:
                return False

            logger.debug(f"Response: {response_json}")
            return True
        except Exception as e:
            logger.debug(f"Could not create assistant run: {e}")
    return False

def create_assistant_event(event: AssistantEventCreate) -> bool:
    if not phi_cli_settings.api_enabled:
        return True

    logger.debug("--o-o-- Creating Assistant Event")
    with api.AuthenticatedClient() as api_client:
        try:
            r: Response = api_client.post(
                ApiRoutes.ASSISTANT_EVENT_CREATE,
                headers={
                    "Authorization": f"Bearer {getenv(PHI_API_KEY_ENV_VAR)}",
                    "PHI-WORKSPACE": f"{getenv(PHI_WS_KEY_ENV_VAR)}",
                },
                json={
                    "event": event.model_dump(exclude_none=True),
                    # "workspace": assistant_workspace.model_dump(exclude_none=True),
                },
            )
            if invalid_response(r):
                return False

            response_json: Union[Dict, List] = r.json()
            if response_json is None:
                return False

            logger.debug(f"Response: {response_json}")
            return True
        except Exception as e:
            logger.debug(f"Could not create assistant event: {e}")
    return False
