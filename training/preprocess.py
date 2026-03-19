import bz2
import re
import html
from pathlib import Path
from tqdm import tqdm
import mwparserfromhell

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def find_dump(raw_dir: Path) -> Path:
    """Locate the single .xml.bz2 dump file in the given directory."""
    dumps = list(raw_dir.glob("*.xml.bz2"))
    if not dumps:
        raise FileNotFoundError(f"No .xml.bz2 file found in {raw_dir}")
    if len(dumps) > 1:
        raise FileNotFoundError(f"Multiple .xml.bz2 files found in {raw_dir}, expected one")
    return dumps[0]


def clean_wikitext(text: str) -> str:
    """Strip wikitext markup from a raw article string using mwparserfromhell, returning plain prose."""
    text = html.unescape(text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)  # ref blocks with content
    text = re.sub(r"<ref[^>]*/?>", "", text)                           # self-closing refs
    text = re.sub(r"<[^>]+>", "", text)                                # any remaining XML/HTML tags
    parsed = mwparserfromhell.parse(text)
    text = parsed.strip_code(normalize=True, collapse=True)
    text = re.sub(r"(?m)^=+.*?=+\s*$", "", text)                      # section headers
    text = re.sub(r"(?m)^thumb\|.*$", "", text)                        # image captions starting with thumb
    text = re.sub(r"(?m)^\w[^|]*\|.*$", "", text)                     # other image/file lines with pipes
    text = re.sub(r"(?m)^Category:.*$", "", text)                      # category tags
    text = re.sub(r"(?m)^\*\s*\d+\s*$", "", text)                     # *04, *08 article markers
    text = re.sub(r"\|\d+px\b[^|]*", "", text)                        # leftover image size tokens
    text = re.sub(r"(?m)^:.*$", "", text)                              # indented lines (talk page artifacts)
    text = re.sub(r"\s+", " ", text)                                   # collapse whitespace
    return text.strip()


def extract(dump_path: Path, output_path: Path) -> None:
    """Parse a bzip2-compressed MediaWiki XML dump and write cleaned article text to a .txt file.
    
    Each line in the output corresponds to one article. Redirects and articles
    shorter than 50 characters are skipped.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    in_page = False
    in_text = False
    current_text = []
    articles_written = 0

    with bz2.open(dump_path, "rt", encoding="utf-8") as f, \
         open(output_path, "w", encoding="utf-8") as out, \
         tqdm(unit=" articles", desc="Extracting") as bar:

        for line in f:
            line = line.strip()

            if "<page>" in line:
                in_page = True
                current_text = []

            if not in_page:
                continue

            if "<text" in line:
                in_text = True
                match = re.search(r"<text[^>]*>(.*)", line)
                if match:
                    current_text.append(match.group(1))
                continue

            if in_text:
                if "</text>" in line:
                    current_text.append(line.replace("</text>", ""))
                    in_text = False
                else:
                    current_text.append(line)

            if "</page>" in line:
                in_page = False
                raw = " ".join(current_text)

                if raw.strip().upper().startswith("#REDIRECT"):
                    continue

                cleaned = clean_wikitext(raw)
                if len(cleaned) < 50:
                    continue

                out.write(cleaned + "\n")
                articles_written += 1
                bar.update(1)

    print(f"Done. {articles_written} articles written to {output_path}")


if __name__ == "__main__":
    dump_path = find_dump(RAW_DIR)
    output_path = PROCESSED_DIR / "corpus.txt"

    if output_path.exists():
        print(f"corpus.txt already exists at {output_path}, skipping.")
    else:
        extract(dump_path, output_path)