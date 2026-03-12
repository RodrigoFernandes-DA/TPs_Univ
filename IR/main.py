from crawler import crawl_wikipedia
from dictionary import DictionaryBuilder
from indexing import DocumentIndexer, QueryIndexer
from similarity import rank_documents

from io_utils import (
    save_urls, load_urls, urls_exist,
    save_dictionary, load_dictionary, dictionary_exist,
)

if __name__ == "__main__":
    test_url = "https://fr.wikipedia.org/wiki/L%27%C3%89vangile_du_monstre_en_spaghettis_volant"
    depth = 2

    print("Testing Wikipedia crawler...")
    print("=" * 50)

    # ---- URL collection ----
    if urls_exist():
        print("Loading URLs from cache...")
        urls = load_urls()
    else:
        print("Crawling Wikipedia...")
        urls = crawl_wikipedia(test_url, depth)
        save_urls(urls)

    print(f"Total URLs discovered: {len(urls)}")

    # ---- Dictionary building ----
    if dictionary_exist():
        print("Loading dictionary from cache...")
        dictionary = load_dictionary()
    else:
        print("Building dictionary...")
        builder = DictionaryBuilder(urls)
        dictionary = builder.build()
        save_dictionary(dictionary)

    print(f"Total unique terms: {len(dictionary)}")


    # ---- Indexer ----
    print("\nIndexing dictionary...")
    indexer = DocumentIndexer(dictionary)


####################################################

    # ---- Querying ----

    doc_vectors = {}
    for url in urls:
        doc_vectors[url] = indexer.index_document(url)

    queries = [
        "religion satire",
        "intelligent design",
        "scientific criticism of religion"
    ]

    query_indexer = QueryIndexer(dictionary)

    schemes = ["boolean", "tf", "wf", "tf_idf", "wf_idf"]

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        query_vec = query_indexer.index_query(query)

        for scheme in schemes:
            print(f"\nTop 10 documents using {scheme}:")
            top_docs = rank_documents(query_vec, doc_vectors, scheme)

            for rank, (url, score) in enumerate(top_docs, start=1):
                print(f"{rank:2d}. {url}  (score={score:.4f})")

