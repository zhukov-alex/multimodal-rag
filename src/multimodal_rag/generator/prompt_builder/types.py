from typing import Protocol
from multimodal_rag.generator.types import GenerateRequest


class PromptBuilder(Protocol):
    def build(self, request: GenerateRequest, model: str) -> dict:
        """
        Returns model-specific input payload
        """
        ...
