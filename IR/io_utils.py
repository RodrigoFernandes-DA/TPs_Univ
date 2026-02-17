import json
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URLS_FILE = DATA_DIR / "urls.json"
DICT_FILE = DATA_DIR / "dictionary.json"
# IND_FILE = DATA_DIR / "indexer.json"


def save_urls(urls) -> None:
    # Convert set → list for JSON
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(urls)), f, ensure_ascii=False, indent=2)


def load_urls():
    # Load list → convert back to set
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_dictionary(dictionary: dict) -> None:
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


def load_dictionary() -> dict:
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# def save_indexer(index: dict) -> None:
#     with open(IND_FILE, "w", encoding="utf-8") as f:
#         json.dump(index, f, ensure_ascii=False, indent=2)


# def load_indexer() -> dict:
#     with open(IND_FILE, "r", encoding="utf-8") as f:
#         return json.load(f)


def urls_exist() -> bool:
    return URLS_FILE.exists()


def dictionary_exist() -> bool:
    return DICT_FILE.exists()


# def indexer_exist() -> bool:
#     return IND_FILE.exists()
