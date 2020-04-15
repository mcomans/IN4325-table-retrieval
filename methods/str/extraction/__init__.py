from methods.types import Table, Query
from enum import Enum
from . import words, entities


class ExtractionMethod(Enum):
    WORDS = "WORDS"
    ENTITY = "ENTITY"
    ENTITY_SPACY = "ENTITY_SPACY"


def extract(table: Table, query: Query,
            method: ExtractionMethod) -> ([str], [str]):
    """Extracts terms from table and query using the given method."""
    if method == ExtractionMethod.ENTITY:
        return (entities.extract_entities(query),
                entities.extract_entities(table))
    elif method == ExtractionMethod.ENTITY_SPACY:
        return (entities.extract_entities(query, use_spacy=True),
                entities.extract_entities(table, use_spacy=True))
    elif method == ExtractionMethod.WORDS:
        return (words.extract_unique_words(query),
                words.extract_unique_words(table))


