import gensim.downloader as api
import re
wv = api.load('word2vec-google-news-300')
print("Loaded word2vec-google-news-300")


def semantic_map(terms: [str]):
    # Split entities that have a _ due to their original form from DBpedia.
    terms = [t for term in terms for t in re.split('[_ ]', term)]
    return [vector for vector in [__get_vector(term) for term in terms] if vector]


def __get_vector(word: str) -> [int]:
    if word in wv:
        return wv[word].tolist()
    if word.capitalize() in wv:
        return wv[word.capitalize()]
    if word.lower() in wv:
        return wv[word.lower()]
    return None
