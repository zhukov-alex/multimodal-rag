from pydantic import BaseModel, Field

from multimodal_rag.generator.types import LLMQueryParams


class LlamaCppParams(BaseModel, LLMQueryParams):
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Controls randomness in output")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    top_k: int = Field(40, ge=0, description="Top-k sampling limit (0 = disable)")
    repeat_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Penalty for repeating tokens")
    n_predict: int = Field(512, ge=1, description="Maximum tokens to generate")
    stop: list[str] = Field(default_factory=list, description="Stop sequences for generation")
    mirostat: int = Field(0, description="Enable Mirostat sampling: 0=off, 1=v1, 2=v2")
    mirostat_tau: float = Field(5.0, description="Target entropy for Mirostat sampling")
    mirostat_eta: float = Field(0.1, description="Adaptation rate for Mirostat")
    grammar: str | None = Field(None, description="Optional grammar constraint (e.g. 'json')")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    logit_bias: dict[int, float] = Field(default_factory=dict, description="Biases applied to specific token IDs")
    penalize_nl: bool = Field(False, description="Apply repeat_penalty to newline tokens")

    @property
    def token_limit(self) -> int:
        return self.n_predict

    def to_payload(self) -> dict:
        payload = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "n_predict": self.n_predict,
            "stop": self.stop,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "penalize_nl": self.penalize_nl,
        }

        if self.seed is not None:
            payload["seed"] = self.seed
        if self.grammar:
            payload["grammar"] = self.grammar
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias

        return payload
