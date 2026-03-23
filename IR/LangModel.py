import random
import re
from collections import defaultdict, Counter
import math
from dictionary import extract_text_from_url


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_language_models(urls, indexer):
    doc_models = {}
    collection_counts = Counter()
    vocabulary = set()
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    total_tokens = 0

    print("\nBuilding document language models...")

    for url in urls:
        doc = indexer.index_document(url)

        text = extract_text_from_url(url) 

        tokens = tokenize(text)
        counts = Counter(tokens)
        
        total_tokens += len(tokens)
        unigram_counts.update(tokens)
        
        for i in range(len(tokens) - 1):
            bigram_counts[tokens[i]][tokens[i + 1]] += 1

        doc_models[url] = {
            "counts": counts,
            "length": len(tokens)
        }

        collection_counts.update(tokens)
        vocabulary.update(tokens)

    # print(tokens)
    V = len(vocabulary)
    
    # Compute prob
    unigram_probs = {
        t: count / total_tokens
        for t, count in unigram_counts.items()
    }

    bigram_probs = {}
    for t_j, next_words in bigram_counts.items():
        total = sum(next_words.values())
        bigram_probs[t_j] = {
            t_i: count / total
            for t_i, count in next_words.items()
        }

    return unigram_probs, bigram_probs, doc_models, V


def sample_from_distribution(distribution):
    words = list(distribution.keys())
    probs = list(distribution.values())
    return random.choices(words, weights=probs, k=1)[0]


def generate_sentence_unigram(unigram_probs, length=10):
    return " ".join(
        sample_from_distribution(unigram_probs)
        for _ in range(length)
    )


def generate_sentence_bigram(unigram_probs, bigram_probs, length=10):
    current = sample_from_distribution(unigram_probs)
    sentence = [current]

    for _ in range(length - 1):
        if current in bigram_probs:
            next_word = sample_from_distribution(bigram_probs[current])
        else:
            next_word = sample_from_distribution(unigram_probs)

        sentence.append(next_word)
        current = next_word

    return " ".join(sentence)


def sentence_probability(sentence, doc_model, V):
    tokens = tokenize(sentence)

    counts = doc_model["counts"]
    doc_len = doc_model["length"]

    log_prob = 0.0 

    for word in tokens:
        prob = (counts[word] + 1) / (doc_len + V)  # Laplace smoothing
        log_prob += math.log(prob)

    return log_prob  # log prob


def rank_documents_by_sentence(sentence, doc_models, V):
    scores = []

    for url, model in doc_models.items():
        score = sentence_probability(sentence, model, V)
        scores.append((url, score))

    # sort by highest probability
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores