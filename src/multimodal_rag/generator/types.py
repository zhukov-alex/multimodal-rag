from typing import Protocol, Literal, AsyncGenerator, Union
from pydantic import BaseModel
from multimodal_rag.document import ScoredItem
from multimodal_rag.generator.params.llamacpp import LlamaCppParams
from multimodal_rag.generator.params.ollama import OllamaParams
from multimodal_rag.generator.params.openai import OpenAIParams


class LLMQueryParams(Protocol):
    def to_payload(self) -> dict:
        ...

    @property
    def token_limit(self) -> int:
        raise NotImplementedError


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    query: str
    context_docs: list[ScoredItem]
    system_prompt: str | None = None
    history: list[ChatMessage] = []
    params: Union[OllamaParams, LlamaCppParams, OpenAIParams]


class Generator(Protocol):
    """
    Interface for generation API with optional streaming.
    """

    async def generate(self, request: GenerateRequest) -> str:
        """
        Generate a full text response.
        """
        ...

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streamed response.
        """
        ...
