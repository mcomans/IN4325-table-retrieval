from functools import reduce
from math import log


def total_idf(terms: [str], doc_set: [[str]]) -> float:
    return sum(__idf(term, doc_set) for term in terms)


def __idf(term: str, doc_set: [[str]]) -> float:
    N = len(doc_set)
    nr_matched_docs = sum(1 for doc_tokens in doc_set if term in doc_tokens)

    return log(N / (1 + nr_matched_docs))


def total_term_frequency(terms: [str], doc_tokens: [str]) -> float:
    terms_lower = [term.lower() for term in terms]
    doc_tokens_lower = [token.lower() for token in doc_tokens]

    return reduce(lambda x, y: x + __term_frequency(y, doc_tokens_lower), terms_lower, 0)


def __term_frequency(term: str, doc_tokens: [str]) -> float:
    """
    Returns the term count in the document tokens
    :param term: Term (expected to be lower case)
    :param doc_tokens: Tokens in document (expected to be lower case)
    :return: Term count in document tokens
    """
    return doc_tokens.count(term)
