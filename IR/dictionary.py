import math
import re
import requests
from bs4 import BeautifulSoup
from collections import defaultdict


def extract_text_from_url(url: str) -> str:
    headers = {
        "User-Agent": "IR-Crawler/1.0 (academic project)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract article
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""

    # Remove tables, navigation boxes, references, etc.
    for tag in content(["script", "style", "noscript", "table", "sup"]):
        tag.decompose()

    text = content.get_text(separator=" ")
    return text


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    return set(token for token in tokens if len(token) > 2)


class DictionaryBuilder:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.N = len(urls)
        self.dictionary = defaultdict(int)

    def build(self) -> dict:
        for url in self.urls:
            text = extract_text_from_url(url)
            if not text:
                continue

            terms_in_doc = tokenize(text)

            for term in terms_in_doc:
                self.dictionary[term] += 1

        return self._compute_idft()

    def _compute_idft(self) -> dict:
        index = {}

        for term, dft in self.dictionary.items():
            idft = math.log(self.N / dft)
            index[term] = {
                "dft": dft,
                "idft": idft
            }

        return index