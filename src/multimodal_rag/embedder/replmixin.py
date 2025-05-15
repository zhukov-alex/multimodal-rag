import os


class ReplicateClientMixin:
    """
    Mixin providing access to Replicate API settings.
    """

    REPLICATE_URL = "https://api.replicate.com/v1"

    @property
    def replicate_token(self) -> str:
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise ValueError("Missing REPLICATE_API_TOKEN environment variable.")
        return token

    @property
    def replicate_url(self) -> str:
        return os.getenv("REPLICATE_BASE_URL", self.REPLICATE_URL) + "/predictions"
