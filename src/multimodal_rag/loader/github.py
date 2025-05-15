import aiohttp
import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator
from base64 import b64decode

from multimodal_rag.document import Document
from multimodal_rag.loader.utils import parse_github_url
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.types import DocumentLoader
from multimodal_rag.loader.directory import DirectoryLoader
from multimodal_rag.log_config import logger

try:
    from tqdm.asyncio import tqdm_asyncio as tqdm
except ImportError:
    from tqdm import tqdm

DEFAULT_MAX_CONNECTIONS = 8


class GitHubRepoLoader(DocumentLoader):
    """
    GitHub loader using GitHub API.
    Extracts all files into a temp folder and delegates loading to DirectoryLoader.
    Supports both directories and single files.
    """

    API_URL = "https://api.github.com"

    def __init__(
        self,
        url: str,
        registry: ReaderRegistry,
        glob: str = "**/*",
        show_progress: bool = False,
    ):
        self.url = url
        self.glob = glob
        self.registry = registry
        self.token = os.getenv("GITHUB_TOKEN")
        self.show_progress = show_progress

    async def load(self) -> list[Document]:
        return [doc async for doc in self.iter_documents()]

    async def iter_documents(self) -> AsyncIterator[Document]:
        info = parse_github_url(self.url)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        connector = aiohttp.TCPConnector(limit=DEFAULT_MAX_CONNECTIONS)

        if not info.is_directory:
            file_url = f"{self.API_URL}/repos/{info.owner}/{info.repo}/contents/{info.path}?ref={info.branch}"
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                target_path = tmp_path / info.path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
                    async with session.get(file_url) as file_resp:
                        file_resp.raise_for_status()
                        data = await file_resp.json()
                        content = data.get("content")
                        encoding = data.get("encoding")
                        if content and encoding == "base64":
                            target_path.write_bytes(b64decode(content))
                        else:
                            raise ValueError("Unsupported or missing content encoding.")

                async for doc in self._load_from_tmp(tmp_path):
                    yield doc
            return

        tree_url = f"{self.API_URL}/repos/{info.owner}/{info.repo}/git/trees/{info.branch}?recursive=1"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
                async with session.get(tree_url) as resp:
                    resp.raise_for_status()
                    tree_data = await resp.json()

                files = [
                    entry["path"] for entry in tree_data.get("tree", [])
                    if entry["type"] == "blob"
                ]

                async def fetch_file(path: str):
                    url = f"{self.API_URL}/repos/{info.owner}/{info.repo}/contents/{path}?ref={info.branch}"
                    logger.debug("Fetching GitHub file", extra={"url": url})
                    async with session.get(url) as file_resp:
                        file_resp.raise_for_status()
                        data = await file_resp.json()
                        content = data.get("content")
                        encoding = data.get("encoding")
                        if content and encoding == "base64":
                            file_path = tmp_path / path
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_bytes(b64decode(content))

                tasks = [fetch_file(p) for p in files]
                iterator = asyncio.as_completed(tasks)
                if self.show_progress:
                    iterator = tqdm(iterator, total=len(tasks), desc="Fetching GitHub files")

                for coro in iterator:
                    await coro

            async for doc in self._load_from_tmp(tmp_path):
                yield doc

    async def _load_from_tmp(self, tmp_path: Path) -> AsyncIterator[Document]:
        dir_loader = DirectoryLoader(
            path=str(tmp_path),
            registry=self.registry,
            glob=self.glob,
            show_progress=self.show_progress,
        )
        async for doc in dir_loader.iter_documents():
            yield doc
