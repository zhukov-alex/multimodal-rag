import asyncio
import shutil
from pathlib import Path

from multimodal_rag.asset_store.types import AssetStore
from multimodal_rag.config.schema import LocalAssetConfig
from multimodal_rag.document import MetaConfig
from multimodal_rag.utils.loader import load_file


class LocalAssetStore(AssetStore):
    def __init__(self, cfg: LocalAssetConfig):
        self.config = cfg

    async def store(
        self,
        project_id: str,
        tmp_path: Path,
        meta: MetaConfig
    ) -> str:
        short_fp = meta.fingerprint[:16]

        name = Path(meta.filename or tmp_path.name).stem
        ext = Path(meta.filename or tmp_path.name).suffix
        file_id = f"{name}_{short_fp}{ext}"

        target_path = Path(self.config.root_dir) / project_id / file_id
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            if not self.config.overwrite:
                raise FileExistsError(f"Asset already exists at {target_path} and overwrite=False")
            target_path.unlink()

        await asyncio.to_thread(shutil.copy2, tmp_path, target_path) # type: ignore
        return f"file://{target_path.resolve()}"

    async def read(self, uri: str) -> bytes:
        return await load_file(uri)

    async def ensure_storage(self, project_id: str) -> None:
        pass
