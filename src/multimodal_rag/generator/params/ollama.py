from pydantic import BaseModel, Field
from typing import Any


class OllamaParams(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Controls randomness in output")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    num_predict: int = Field(512, ge=1, description="Maximum tokens to generate")
    repeat_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Penalty for repeating tokens")
    repeat_last_n: int = Field(64, ge=0, description="How many tokens to apply repetition penalty to")
    stop: list[str] = Field(default_factory=list, description="Stop sequences for generation")
    mirostat: int = Field(0, description="Enable Mirostat sampling: 0=off, 1=v1, 2=v2")
    mirostat_tau: float = Field(5.0, description="Target entropy for Mirostat sampling")
    mirostat_eta: float = Field(0.1, description="Adaptation rate for Mirostat sampling")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Minimum probability threshold for tokens")
    num_ctx: int = Field(2048, ge=128, description="Context window size for the model")
    format: str | dict[str, Any] | None = Field(None, description="Optional structured output format (e.g. JSON schema)")
    raw: bool = Field(False, description="If true, disables post-processing of the output")

    @property
    def token_limit(self) -> int:
        return self.num_predict

    def to_payload(self) -> dict:
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
            "stop": self.stop,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "min_p": self.min_p,
            "repeat_last_n": self.repeat_last_n,
            "num_ctx": self.num_ctx,
            "raw": self.raw
        }

        if self.format:
            options["format"] = self.format

        return {"options": options}
