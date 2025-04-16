from pydantic import BaseModel, Field

from a2a.common.types import (
    JSONRPCResponse,
    ContentTypeNotSupportedError,
    UnsupportedOperationError,
)
from typing import List, Literal


def are_modalities_compatible(
    server_output_modes: List[str], client_output_modes: List[str]
):
    """Modalities are compatible if they are both non-empty
    and there is at least one common element."""
    if client_output_modes is None or len(client_output_modes) == 0:
        return True

    if server_output_modes is None or len(server_output_modes) == 0:
        return True

    return any(x in server_output_modes for x in client_output_modes)


def new_incompatible_types_error(request_id):
    return JSONRPCResponse(id=request_id, error=ContentTypeNotSupportedError())


def new_not_implemented_error(request_id):
    return JSONRPCResponse(id=request_id, error=UnsupportedOperationError())


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = Field(default="input_required", description="response status")
    message: str = Field(description="Your output message")