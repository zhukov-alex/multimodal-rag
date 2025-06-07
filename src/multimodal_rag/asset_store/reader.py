import asyncio

from multimodal_rag.asset_store.types import AssetStore
from multimodal_rag.utils.loader import image_bytes_to_base64
from multimodal_rag.log_config import logger


class AssetReaderService:
    def __init__(self, stores: dict[str, AssetStore]):
        self.stores = stores

    async def read(self, storage_type: str, uri: str) -> bytes:
        if storage_type not in self.stores:
            raise ValueError(f"No reader for storage_type: {storage_type}")
        return await self.stores[storage_type].read(uri)

    async def read_image_base64(self, storage_type: str, uri: str, format: str = "PNG") -> str | None:
        try:
            image_bytes = await self.read(storage_type, uri)
            return await asyncio.to_thread(image_bytes_to_base64, image_bytes, format)
        except Exception as e:
            logger.warning(f"Failed to encode image from {uri}: {e}")
            return None
