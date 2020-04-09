from nltk.tokenize import word_tokenize
from methods.str.types import Table


def extract_entities(terms: [str]) -> [str]:
    """Extract named entities from the given terms."""
    tokenized = [word_tokenize(term) for term in terms]
    return [term for term in tokenized if term.label == "NE"]


def __find_core_column(table: Table):
    """Finds the core column of a table using entity analysis.
    The core column is selected as the column with the highest ratio of
    entities/cells.
    """
    # TODO: Implementation.
    pass


def __take_top_k(entities: [str], k: int = 10):
    """Takes the top-k entities using Mixture of Language Models approach.
    The default value is 10 as this was the default value set by Zhang and
    Balog.
    """
    # TODO: Implementation.
    pass
