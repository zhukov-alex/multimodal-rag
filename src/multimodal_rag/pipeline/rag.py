from pydantic import BaseModel, Field
from pathlib import Path

from multimodal_rag.config.schema import RAGConfig
from multimodal_rag.config.factory import (
    create_text_embedder,
    create_image_embedder,
    create_storage_client,
    create_asset_stores,
    create_reranker,
    create_generator,
    parse_llm_params,
)
from multimodal_rag.embedder.service import EmbedderService
from multimodal_rag.asset_store.reader import AssetReaderService
from multimodal_rag.retriever.service import MultiModalRetriever
from multimodal_rag.retriever.types import SearchByText, SearchByImage
from multimodal_rag.generator.service import GeneratorService
from multimodal_rag.generator.types import Generator, GenerateRequest
from multimodal_rag.log_config import logger
from multimodal_rag.utils.loader import load_image_base64
from multimodal_rag.utils.timing import log_duration


class RAGRequest(BaseModel):
    query: str | None = None
    ask: str | None = None
    image_path: Path | None = None
    modality_top_k: dict[str, int] = Field(default_factory=lambda: {"text": 5, "image": 5})
    system_prompt: str = "You're a helpful assistant."
    llm_params: dict[str, object]


async def run_rag_pipeline(
    config: RAGConfig,
    request: RAGRequest,
    project_id: str,
    stream: bool,
):
    logger.info("Starting RAG pipeline", extra={"project_id": project_id, "stream": stream})

    # --- Init services ---
    text_embedder = create_text_embedder(config.embedding.text)
    image_embedder = create_image_embedder(config.embedding.image) if config.embedding.image else None
    embedder_service = EmbedderService(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        batch_size=config.embedding.batch_size or 64,
    )

    asset_reader = AssetReaderService(stores=create_asset_stores(config.asset_store))
    storage = create_storage_client(config.storaging)
    reranker = create_reranker(config.reranking) if config.reranking else None
    retriever = MultiModalRetriever(
        embedder=embedder_service,
        storage=storage,
        asset_reader=asset_reader,
        reranker=reranker
    )
    generator: Generator = create_generator(config.generation)
    generator_service = GeneratorService(generator)

    try:
        # --- Retrieve ---
        context_docs = []
        if request.query:
            async with log_duration("retrieve_by_text", query=request.query, top_k=request.modality_top_k):
                search_request = SearchByText(
                    query=request.query,
                    project_id=project_id,
                    modality_top_k=request.modality_top_k,
                    filters={},
                    search_type="embedding",
                )
                context_docs = await retriever.retrieve_by_text(search_request)

        elif request.image_path:
            async with log_duration("retrieve_by_image", image_path=str(request.image_path), top_k=request.modality_top_k):
                img_b64 = await load_image_base64(str(request.image_path))
                search_request = SearchByImage(
                    img_b64=img_b64,
                    project_id=project_id,
                    top_k=request.modality_top_k.get("image", 0),
                    filters={},
                )
                context_docs = await retriever.retrieve_by_image(search_request)

        # --- Build generation request ---
        question = request.ask or request.query
        if not question:
            logger.error("Neither query nor ask was provided")
            raise ValueError("You must provide either 'ask' or 'query'.")

        history = await load_history(project_id)
        llm_params = parse_llm_params(config.generation.type, request.llm_params)

        gen_request = GenerateRequest(
            query=question,
            context_docs=context_docs,
            history=history,
            system_prompt=request.system_prompt,
            params=llm_params,
        )

        logger.debug("Generated request", extra={"request": gen_request})

        # --- Generate ---
        if stream:
            print("\n[ANSWER]\n", end="", flush=True)
            response = ""
            async with log_duration("stream_response", tokens=llm_params.token_limit):
                try:
                    async for chunk in generator_service.generate_stream(gen_request):
                        response += chunk
                        print(chunk, end="", flush=True)
                except Exception as e:
                    logger.exception("Streaming generation failed", extra={"error": str(e)})
                    print("\n[STREAM ERROR]", flush=True)

            await save_history(project_id, question, response)

        else:
            async with log_duration("generate_response", tokens=llm_params.token_limit):
                response = await generator_service.generate(gen_request)

            logger.info("Generated response", extra={"length": len(response), "response": response})
            print("\n[ANSWER]\n", response)

            await save_history(project_id, question, response)

    except Exception as e:
        logger.exception("RAG pipeline failed", extra={"error": str(e)})
        raise
    finally:
        await storage.close()


# Stubs
async def load_history(project_id: str) -> list[dict[str, str]]:
    logger.debug(f"Loading conversation history for project: {project_id}")
    return []

async def save_history(project_id: str, query: str, response: str) -> None:
    logger.debug(f"Storing conversation history for project: {project_id}, query: {query[:100]}..., response length: {len(response)}")
