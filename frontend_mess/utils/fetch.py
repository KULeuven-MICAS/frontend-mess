from pathlib import Path
from urllib.request import urlretrieve


CACHE_DIR = Path.home() / ".cache" / "frontend-mess" / "models"


def _to_raw_github(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def fetch_model(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / Path(url).name
    if not dst.exists():
        urlretrieve(_to_raw_github(url), dst)
    return dst
