from pydantic import BaseModel, Field

from multimodal_rag.generator.types import LLMQueryParams


class OpenAIParams(BaseModel, LLMQueryParams):
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Controls randomness in output")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    max_tokens: int = Field(512, ge=1, description="Maximum tokens to generate")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Penalty for using new tokens")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Penalty for repeating tokens")
    stop: str | list[str] = Field(default_factory=list, description="Stop sequences for generation")
    logit_bias: dict[str, float] = Field(default_factory=dict, description="Bias for specific token IDs")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    response_format: str | None = Field(None, description="Optional output format (e.g. 'json')")

    @property
    def token_limit(self) -> int:
        return self.max_tokens

    def to_payload(self) -> dict:
        payload = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias
        if self.response_format:
            payload["response_format"] = self.response_format

        return payload
