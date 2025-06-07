from pathlib import Path
import tempfile
import shutil

_tmp_dirs: set[Path] = set()


def make_tmp_dir(*, prefix: str = "tmp", suffix: str = "", dir: str | None = None) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir))
    _tmp_dirs.add(path)
    return path


def get_tmp_dirs() -> set[Path]:
    return _tmp_dirs.copy()


def cleanup_tmp_dirs() -> None:
    for path in get_tmp_dirs():
        shutil.rmtree(path, ignore_errors=True)
    _tmp_dirs.clear()
