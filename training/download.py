import requests
from tqdm import tqdm
from pathlib import Path

DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
RAW_DIR = Path(__file__).parent / "data" / "raw"

def download_dump(url: str = DUMP_URL, dest_dir: Path = RAW_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"Dump already exists at {dest_path}, skipping download.")
        return dest_path

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved to {dest_path}")
    return dest_path

if __name__ == "__main__":
    download_dump()