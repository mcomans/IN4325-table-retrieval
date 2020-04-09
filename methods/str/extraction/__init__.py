from methods.str.types import Table, Query, TermVector
from enum import Enum
from . import words, entities


class ExtractionMethod(Enum):
    WORDS = "WORDS"
    ENTITY = "ENTITY"


def extract(terms: [str],
            method: ExtractionMethod) -> TermVector:
    """Extracts terms from the table and query given the method."""
    if method == ExtractionMethod.ENTITY:
        return entities.extract_entities(terms)
    elif method == ExtractionMethod.WORDS:
        return words.extract_unique_words(terms)
