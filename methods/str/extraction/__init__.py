from methods.str.types import Table, Query, TermVector
from enum import Enum
from . import words, entities


class ExtractionMethod(Enum):
    WORDS = "WORDS"
    ENTITY = "ENTITY"


def extract(table: Table, query: Query,
            method: ExtractionMethod) -> (TermVector, TermVector):
    """Extracts terms from table and query using the given method."""
    if method == ExtractionMethod.ENTITY:
        return (entities.extract_entities(query),
                entities.extract_entities(table))
    elif method == ExtractionMethod.WORDS:
        return (words.extract_unique_words(query),
                words.extract_unique_words(table))


