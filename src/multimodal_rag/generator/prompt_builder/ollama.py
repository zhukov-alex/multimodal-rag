import os.path

from multimodal_rag.generator.types import GenerateRequest
from multimodal_rag.generator.prompt_builder.types import PromptBuilder


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "../templates/ollama_multimodal.tpl")


class OllamaPromptBuilder(PromptBuilder):
    def build(self, request: GenerateRequest, _: str) -> dict:
        history_part = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in request.history)

        image_prompts = []
        if request.context_docs:
            for doc in request.context_docs:
                if doc.modality == "image" and doc.caption:
                    image_prompts.append(f"Image: {doc.caption}")
                elif doc.modality == "text":
                    image_prompts.append(f"{doc.content}")

        prompt = "\n".join(filter(None, [
            "\n".join(image_prompts),
            history_part,
            f"User: {request.query}",
            "Assistant:"
        ]))

        images = [doc.image_base64 for doc in request.context_docs if doc.modality == "image" and doc.image_base64]

        payload = {
            "prompt": prompt,
            "system": request.system_prompt or "",
            "images": images,
            **request.params.to_payload()
        }

        if images:
            try:
                with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                    payload["template"] = f.read()
            except Exception as e:
                from multimodal_rag.log_config import logger
                logger.warning(f"Failed to load Ollama template: {e}")

        return payload
