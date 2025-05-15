from pathlib import Path
from urllib.parse import urlparse
from typing import NamedTuple
from multimodal_rag.constants import SUPPORTED_ARCHIVES, KNOWN_BUT_UNSUPPORTED


class GitRepoInfo(NamedTuple):
    owner: str
    repo: str
    branch: str
    path: str
    is_directory: bool


def is_archive(path: Path) -> bool:
    suffix = "".join(path.suffixes)
    return suffix in (SUPPORTED_ARCHIVES | KNOWN_BUT_UNSUPPORTED)


def is_github_url(path: str) -> bool:
    if path.startswith("git@github.com:"):
        return True
    try:
        parsed = urlparse(path)
        return parsed.scheme in {"http", "https", "git"} and parsed.netloc == "github.com"
    except Exception:
        return False


def parse_github_url(url: str) -> GitRepoInfo:
    """
    Parses a GitHub URL and returns repo metadata. Supports:
    - SSH git@github.com:user/repo.git
    - HTTPS https://github.com/user/repo.git
    - UI URLs with /tree/branch or /blob/branch
    """
    url = url.strip()

    # SSH git@github.com:user/repo.git
    if url.startswith("git@github.com:"):
        path = url.split("git@github.com:")[1].rstrip(".git")
        parts = path.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid SSH GitHub URL")
        return GitRepoInfo(owner=parts[0], repo=parts[1], branch="main", path=".", is_directory=True)

    # HTTPS Git clone URL
    if url.startswith("https://github.com") and url.endswith(".git"):
        parts = urlparse(url).path.strip("/").rstrip(".git").split("/")
        if len(parts) != 2:
            raise ValueError("Invalid clone URL")
        return GitRepoInfo(parts[0], parts[1], "main", ".", True)

    # UI URL
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc != "github.com" or len(parts) < 2:
        raise ValueError("Invalid GitHub URL")

    owner, repo = parts[0], parts[1].replace(".git", "")
    branch, subpath, is_dir = "main", ".", True

    if "tree" in parts:
        idx = parts.index("tree")
        branch = parts[idx + 1] if len(parts) > idx + 1 else "main"
        subpath = "/".join(parts[idx + 2:]) or "."
    elif "blob" in parts:
        idx = parts.index("blob")
        branch = parts[idx + 1] if len(parts) > idx + 1 else "main"
        subpath = "/".join(parts[idx + 2:]) or "."
        is_dir = False

    return GitRepoInfo(owner, repo, branch, subpath, is_dir)
