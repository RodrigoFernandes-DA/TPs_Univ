import math
import re
from collections import Counter
from dictionary import extract_text_from_url

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())

class DocumentIndexer:
    def __init__(self, dictionary: dict):
        """
        dictionary structure:
        {
            term: {
                "dft": int,
                "idft": float
            }
        }
        """
        self.dictionary = dictionary
        self.terms = list(dictionary.keys())

    def index_document(self, url: str) -> dict:
        text = extract_text_from_url(url)
        tokens = tokenize(text)

        tf = Counter(tokens)

        return {
            "boolean": self._boolean_model(tf),
            "tf": self._tf_model(tf),
            "wf": self._wf_model(tf),
            "tf_idf": self._tf_idf_model(tf),
            "wf_idf": self._wf_idf_model(tf),
        }

    def _boolean_model(self, tf: Counter) -> dict:
        return {
            term: 1 if tf.get(term, 0) > 0 else 0
            for term in self.terms
        }

    def _tf_model(self, tf: Counter) -> dict:
        return {
            term: tf.get(term, 0)
            for term in self.terms
        }

    def _wf_model(self, tf: Counter) -> dict:
        return {
            term: (1 + math.log(tf[term])) if tf.get(term, 0) > 0 else 0
            for term in self.terms
        }

    def _tf_idf_model(self, tf: Counter) -> dict:
        return {
            term: tf.get(term, 0) * self.dictionary[term]["idft"]
            for term in self.terms
        }

    def _wf_idf_model(self, tf: Counter) -> dict:
        return {
            term: ((1 + math.log(tf[term])) * self.dictionary[term]["idft"])
            if tf.get(term, 0) > 0 else 0
            for term in self.terms
        }


# output structure:
# {
#   "boolean": {term → 0/1},
#   "tf": {term → tf},
#   "wf": {term → wf},
#   "tf_idf": {term → weight},
#   "wf_idf": {term → weight}
# }


class QueryIndexer(DocumentIndexer):
    def index_query(self, query: str) -> dict:
        tokens = tokenize(query)
        tf = Counter(tokens)

        return {
            "boolean": self._boolean_model(tf),
            "tf": self._tf_model(tf),
            "wf": self._wf_model(tf),
            "tf_idf": self._tf_idf_model(tf),
            "wf_idf": self._wf_idf_model(tf),
        }
