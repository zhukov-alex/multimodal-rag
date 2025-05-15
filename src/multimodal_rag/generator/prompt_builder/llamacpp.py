from multimodal_rag.generator.types import GenerateRequest
from multimodal_rag.generator.prompt_builder.types import PromptBuilder


class LlamaCppPromptBuilder(PromptBuilder):
    def build(self, request: GenerateRequest, model: str) -> dict:
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for doc in request.context_docs or []:
            if doc.modality == "image" and doc.image_base64:
                content = []
                if doc.caption:
                    content.append({"type": "text", "text": doc.caption})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{doc.image_base64}"}
                })
                messages.append({"role": "user", "content": content})
            elif doc.modality == "text":
                messages.append({"role": "system", "content": doc.content})

        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": request.query})

        return {
            "model": model,
            "messages": messages,
            **request.params.to_payload()
        }
