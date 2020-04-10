from nltk.tokenize import word_tokenize
from methods.str.types import Table


def extract_entities(terms: [str]) -> [str]:
    """Extract named entities from the given terms."""
    tokenized = [word_tokenize(term) for term in terms]
    return [term for term in tokenized if term.label == "NE"]


def __find_core_column(table: Table) -> int:
    """Finds the core column of a table using entity analysis.
    The core column is selected as the column with the highest ratio of
    entities/cells.
    """
    scores = []
    for col in range(table.numCols):
        col_data = [row[col] for row in table.rows()]
        scores.append((col, len(extract_entities(col_data)) / table.numCols))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0]


def __take_top_k(entities: [str], k: int = 10):
    """Takes the top-k entities using Mixture of Language Models approach.
    The default value is 10 as this was the default value set by Zhang and
    Balog.
    """
    # TODO: Implementation.
    pass
