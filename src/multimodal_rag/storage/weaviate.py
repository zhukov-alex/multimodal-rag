from weaviate import (
    WeaviateAsyncClient,
    use_async_with_local,
    use_async_with_weaviate_cloud,
    use_async_with_embedded,
)
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.filters import Filter
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery

from multimodal_rag.document import Document, ScoredChunk, Chunk
from multimodal_rag.config.schema import WeaviateConnectionConfig
from multimodal_rag.storage.types import AggregateFilter, StorageClient
from multimodal_rag.log_config import logger
from multimodal_rag.storage.utils import normalize_model_name


class WeaviateClient(StorageClient):
    def __init__(self, config: WeaviateConnectionConfig):
        self.config = config
        self.client: WeaviateAsyncClient | None = None

    async def get_connection(self) -> WeaviateAsyncClient:
        if self.client is None or not await self.client.is_ready():
            await self._connect()
        return self.client

    async def _connect(self) -> None:
        additional_config = AdditionalConfig(timeout=Timeout(init=60, query=300, insert=300))

        match self.config.deployment:
            case "cloud":
                self.client = use_async_with_weaviate_cloud(
                    cluster_url=self.config.url,
                    auth_credentials=self._get_auth(),
                    additional_config=additional_config,
                )
            case "local":
                self.client = use_async_with_local(
                    host=f"{'https' if self.config.secure else 'http'}://{self._extract_host()}",
                    port=self.config.port,
                    auth_credentials=self._get_auth(),
                    additional_config=additional_config,
                )
            case "embedded":
                self.client = use_async_with_embedded(
                    additional_config=additional_config,
                )
            case _:
                raise ValueError(f"Unsupported deployment type: {self.config.deployment}")

        await self.client.connect()
        logger.debug("Connected to Weaviate", extra={"deployment": self.config.deployment, "url": self.config.url})

    async def create_embedding_collection(self, name: str | None, emb_model_name: str, dim: int, distance: str = "cosine") -> str:
        client = await self.get_connection()
        norm_model = normalize_model_name(emb_model_name)
        collection_name = f"{name}_embedding_{norm_model}" if name else f"embedding_{norm_model}"

        if await client.collections.exists(collection_name):
            logger.debug("Embedding collection already exists", extra={"name": collection_name})
            return collection_name

        await client.collections.create_from_dict({
            "class": collection_name,
            "vectorizer": "none",
            "vectorIndexConfig": {
                "distance": distance,
                "dimensions": dim
            },
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["text"]},
                {"name": "doc_uuid", "dataType": ["text"]}
            ],
            "autoSchema": False
        })

        logger.debug("Created embedding collection", extra={"name": collection_name, "dim": dim})
        return collection_name

    async def create_document_collection(self, name: str) -> str:
        client = await self.get_connection()
        collection_name = f"{name}_documents"

        if await client.collections.exists(collection_name):
            logger.debug("Document collection already exists", extra={"name": collection_name})
            return collection_name

        await client.collections.create_from_dict({
            "class": collection_name,
            "vectorizer": "none",
            "properties": [
                {"name": "storage_type", "dataType": ["text"]},
                {"name": "asset_uri", "dataType": ["text"]},
                {"name": "file_reader", "dataType": ["text"]},
                {"name": "parsed_format", "dataType": ["text"]},
                {"name": "labels", "dataType": ["text[]"]},
                {"name": "filename", "dataType": ["text"]},
                {"name": "size_bytes", "dataType": ["int"]},
                {"name": "last_modified", "dataType": ["int"]},
                {"name": "fingerprint", "dataType": ["text"]},
                {"name": "mime", "dataType": ["text"]},
            ],
            "autoSchema": False
        })

        logger.debug("Created document collection", extra={"name": collection_name})
        return collection_name

    async def insert_documents(self, documents: list[Document], collection_name: str) -> None:
        client = await self.get_connection()
        collection = client.collections.get(collection_name)
        objects = [doc.to_json() for doc in documents]
        await collection.data.insert_many(objects)
        logger.debug("Inserted documents", extra={"collection": collection_name, "count": len(objects)})

    async def insert_chunks(self, documents: list[Document], collection_name: str) -> None:
        client = await self.get_connection()
        collection = client.collections.get(collection_name)

        objects, vectors = [], []
        for doc in documents:
            for chunk in doc.chunks:
                objects.append({
                    "content": chunk.content,
                    "chunk_id": str(chunk.chunk_id),
                    "doc_uuid": doc.uuid
                })
                vectors.append(chunk.embedding)

        await collection.data.insert_many(objects=objects, vectors=vectors)
        logger.debug("Inserted chunks", extra={"collection": collection_name, "count": len(objects)})

    async def delete_by_ids(self, collection_name: str, field: str, ids: list[str]) -> None:
        client = await self.get_connection()
        collection = client.collections.get(collection_name)
        for _id in ids:
            await collection.data.delete_many(where=Filter.by_property(field).equal(_id))
        logger.debug("Deleted by ids", extra={"collection": collection_name, "field": field, "count": len(ids)})

    async def aggregate_total_count(self, collection_name: str, filter_by: AggregateFilter) -> int:
        client = await self.get_connection()
        collection = client.collections.get(collection_name)
        filters = Filter.by_property(filter_by.field).equal(filter_by.value)
        response = await collection.aggregate.over_all(filters=filters, total_count=True)
        logger.debug("Aggregated total count", extra={"collection": collection_name, "filter": filter_by.dict(), "total": response.total_count})
        return response.total_count

    async def query_by_filter(self, collection_name: str, filters: dict) -> list[dict]:
        client = await self.get_connection()
        collection = client.collections.get(collection_name)
        wv_filters = self.build_filter(filters.get("and", []))
        results = await collection.query.fetch_objects(filters=wv_filters)
        return [obj.properties for obj in results.objects]

    async def query_by_text(self, query: str, filters: dict | None = None) -> list[dict]:
        client = await self.get_connection()
        collection = client.collections.get(self.config.class_name)
        wv_filters = self.build_filter(filters.get("and", [])) if filters else None
        results = await collection.query.bm25(query=query, limit=10, filters=wv_filters)
        logger.debug("Performed text query", extra={"query": query, "results": len(results.objects)})
        return [obj.properties for obj in results.objects]

    async def query_by_vector(self, vector: list[float], collection_name: str, filters: dict | None = None, top_k: int = 10) -> list[ScoredChunk]:
        collection = self.client.collections.get(collection_name)
        wv_filters = self.build_filter(filters.get("and", [])) if filters else None
        results = await collection.query.near_vector(
            near_vector=vector,
            where=wv_filters,
            limit=top_k,
            return_metadata=MetadataQuery(score=True, explain_score=False)
        )
        return self._build_scored_chunks(results.objects)

    async def hybrid_chunks(
            self, query: str, vector: list[float], collection_name: str, limit: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        collection = self.client.collections.get(collection_name)
        wv_filters = self.build_filter(filters.get("and", [])) if filters else None
        results = await collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=0.5,
            limit=limit,
            where=wv_filters,
            return_metadata=MetadataQuery(score=True, explain_score=False)
        )
        return self._build_scored_chunks(results.objects)

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            logger.debug("Closed Weaviate connection")

    @staticmethod
    def _build_scored_chunks(objects) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=int(obj.properties["chunk_id"]),
                    content=obj.properties["content"],
                ),
                score=obj.metadata.score
            ) for obj in objects
        ]

    def _get_auth(self):
        if self.config.api_key:
            return AuthApiKey(self.config.api_key)
        return None

    def _extract_host(self) -> str:
        if self.config.url:
            return self.config.url.replace("https://", "").replace("http://", "")
        return "localhost"

    @staticmethod
    def build_filter(filters: list[dict]):
        """
        Accepts a list of filters and combines them with AND (&).
        Each filter must be a dict with keys: field, operator, value
        """

        if not filters:
            raise ValueError("Filter list cannot be empty")

        def _single(f: dict):
            field = f["field"]
            op = f["operator"].lower()
            value = f.get("value")
            prop = Filter.by_property(field)

            match op:
                case "equal":
                    return prop.equal(value)
                case "not_equal":
                    return prop.not_equal(value)
                case "like":
                    return prop.like(value)
                case "greater_than":
                    return prop.greater_than(value)
                case "less_than":
                    return prop.less_than(value)
                case "contains_any":
                    return prop.contains_any(value)
                case "contains_all":
                    return prop.contains_all(value)
                case _:
                    raise ValueError(f"Unsupported operator: {op}")

        filters_built = [_single(f) for f in filters]
        combined = filters_built[0]
        for f in filters_built[1:]:
            combined &= f
        return combined
