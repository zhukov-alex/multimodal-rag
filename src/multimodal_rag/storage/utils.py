import re


def normalize_model_name(name: str):
    return re.sub(r'[^a-zA-Z0-9]+', '_', name)
