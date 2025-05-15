import requests

from multimodal_rag.log_config import logger


def resolve_model_name(name: str) -> str | None:
    hf = _lazy_import_hf()
    if hf is None:
        return None

    api = hf.HfApi()
    if "/" in name:
        return name

    try:
        results = api.list_models(search=name)
        for model in results:
            if model.modelId.endswith(f"/{name}"):
                return model.modelId
    except Exception as e:
        logger.exception(f"Model name resolution failed for '{name}': {e}")
    return None


def get_model_config_value(model_name: str, key: str = "") -> dict:
    hf = _lazy_import_hf()
    if hf is None:
        return {"model": None, key or "config": None, "error": "feature unavailable"}

    resolved = resolve_model_name(model_name)
    if not resolved:
        logger.error(f"Model '{model_name}' not found during resolution.")
        return {"model": None, key or "config": None, "error": "model not found"}

    url = f"https://huggingface.co/{resolved}/resolve/main/config.json"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        config = resp.json()
    except Exception as e:
        logger.exception(f"Failed to fetch config for model '{resolved}': {e}")
        return {"model": resolved, key or "config": None, "error": f"failed to load config: {e}"}

    if key:
        if key in config:
            return {"model": resolved, key: config.get(key), "error": None}
        else:
            logger.warning(f"Key '{key}' not found in config for model '{resolved}'")
            return {"model": resolved, key: None, "error": "key not found"}
    else:
        return {"model": resolved, "config": config, "error": None}


def _lazy_import_hf():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping HuggingFace-related features.")
        return None
