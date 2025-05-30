import asyncio
import shutil
from pathlib import Path

from multimodal_rag.asset_store.types import AssetStore
from multimodal_rag.config.schema import LocalAssetConfig
from multimodal_rag.document import MetaConfig


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

        await asyncio.to_thread(shutil.move, str(tmp_path), str(target_path))
        return f"file://{target_path.resolve()}"
