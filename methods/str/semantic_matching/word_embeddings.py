import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


def semantic_map(terms: [str]):
    return [vector for vector in [__get_vector(term) for term in terms] if vector]


def __get_vector(word: str) -> [int]:
    try:
        return wv[word].toList()
    except KeyError:
        return None
