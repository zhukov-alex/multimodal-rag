import aiohttp
import asyncio
import os
from pathlib import Path
from base64 import b64decode
from aiohttp import ClientError
from asyncio import TimeoutError

from multimodal_rag.loader.utils import parse_github_url
from multimodal_rag.loader.reader.registry import ReaderRegistry
from multimodal_rag.loader.types import DocumentLoader, LoadResult
from multimodal_rag.utils.retry import backoff
from multimodal_rag.log_config import logger
from multimodal_rag.utils.temp_dirs import make_tmp_dir

try:
    from tqdm.asyncio import tqdm_asyncio as tqdm
except ImportError:
    from tqdm import tqdm

DEFAULT_MAX_CONNECTIONS = 8


class GitHubRepoLoader(DocumentLoader):
    """
    GitHub loader using GitHub API.
    Extracts all files into a temp folder and returns for further processing.
    """

    API_URL = "https://api.github.com"

    def __init__(
        self,
        registry: ReaderRegistry,
        show_progress: bool = False,
    ):
        self.registry = registry
        self.token = os.getenv("GITHUB_TOKEN")
        self.show_progress = show_progress

    async def load(self, source: str, _: str | None = None) -> LoadResult:
        info = parse_github_url(source)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        connector = aiohttp.TCPConnector(limit=DEFAULT_MAX_CONNECTIONS)
        tmp_path = make_tmp_dir()

        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            if not info.is_directory:
                await self._fetch_and_store_file(session, info.path, info, tmp_path)
            else:
                await self._fetch_repository_tree(info, tmp_path, session)

        return LoadResult(documents=[], next_sources=[str(tmp_path)])

    @backoff(exception=(ClientError, TimeoutError))
    async def _fetch_file_metadata(self, session: aiohttp.ClientSession, url: str) -> dict:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _fetch_and_store_file(self, session: aiohttp.ClientSession, path: str, info, tmp_path: Path):
        url = f"{self.API_URL}/repos/{info.owner}/{info.repo}/contents/{path}?ref={info.branch}"
        logger.debug("Fetching GitHub file", extra={"url": url})
        data = await self._fetch_file_metadata(session, url)
        content = data.get("content")
        encoding = data.get("encoding")
        if content and encoding == "base64":
            file_path = tmp_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(file_path.write_bytes, b64decode(content))
        else:
            raise ValueError("Unsupported or missing content encoding.")

    async def _fetch_repository_tree(self, info, tmp_path: Path, session: aiohttp.ClientSession):
        tree_url = f"{self.API_URL}/repos/{info.owner}/{info.repo}/git/trees/{info.branch}?recursive=1"
        tree_data = await self._fetch_file_metadata(session, tree_url)

        files = [
            entry["path"] for entry in tree_data.get("tree", [])
            if entry["type"] == "blob"
        ]

        tasks = [
            self._fetch_and_store_file(session, path, info, tmp_path)
            for path in files
        ]

        iterator = asyncio.as_completed(tasks)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Fetching GitHub files")

        for coro in iterator:
            await coro
