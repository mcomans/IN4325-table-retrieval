from methods.types import Table, Query
from methods.str import dbpedia_api
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(input, use_spacy=False) -> [str]:
    """Extract named entities from the given terms."""
    terms = []
    if isinstance(input, Table):
        terms = __extract_table_terms(input, use_spacy)
    elif isinstance(input, Query):
        terms = [input.query]
    elif isinstance(input, list):
        terms = input
    else:
        terms = [input]
    assert terms is not None

    if use_spacy:
        entities = [__spacy_entities(term) for term in terms]
    else:
        entities = [dbpedia_api.extract_entities(term) for term in terms]

    return [ent for list in entities for ent in list]


def __spacy_entities(term: str) -> [str]:
    doc = nlp(term)
    return [ent.text for ent in doc.ents]


def __extract_table_terms(table: Table, use_spacy=False) -> [str]:
    """Uses some aspects of the Table input to retrieve the terms relevant as
    defined by Zhang and Balog. For entity extraction this is the core
    column, page title and table caption."""
    return __find_core_column(table, use_spacy) + [table.page_title, table.caption]


def __find_core_column(table: Table, use_spacy=False) -> [str]:
    """Returns the values of the core column of a table using entity analysis.
    The core column is selected as the column with the highest ratio of
    entities/cells.
    """
    scores = []
    for col in range(table.num_cols):
        col_data = [row[col] for row in table.rows()]
        scores.append((col, col_data, len(extract_entities(col_data, use_spacy)) /
                       table.num_cols))
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
    if len(sorted_scores) > 0:
        return sorted_scores[0][1]
    else:
        return []


def __take_top_k(entities: [str], k: int = 10):
    """Takes the top-k entities using Mixture of Language Models approach.
    The default value is 10 as this was the default value set by Zhang and
    Balog.
    """
    # TODO: Implementation.
    pass
