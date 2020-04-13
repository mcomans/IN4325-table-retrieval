import spacy
from methods.types import Table, Query

nlp = spacy.load('en_core_web_sm')


def extract_entities(input) -> [str]:
    """Extract named entities from the given terms."""
    terms = []
    if isinstance(input, Table):
        terms = __extract_table_terms(input)
    elif isinstance(input, Query):
        terms = [input.query]
    elif isinstance(input, list):
        terms = input
    else:
        terms = [input]
    assert terms is not None
    entities = [nlp(term).ents for term in terms]
    return [ent for list in entities for ent in list]


def __extract_table_terms(table: Table) -> [str]:
    """Uses some aspects of the Table input to retrieve the terms relevant as
    defined by Zhang and Balog. For entity extraction this is the core
    column, page title and table caption."""
    return __find_core_column(table) + [table.page_title, table.caption]


def __find_core_column(table: Table) -> [str]:
    """Returns the values of the core column of a table using entity analysis.
    The core column is selected as the column with the highest ratio of
    entities/cells.
    """
    scores = []
    for col in range(table.num_cols):
        col_data = [row[col] for row in table.rows()]
        scores.append((col, col_data, len(extract_entities(col_data)) /
                       table.num_cols))
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
    return sorted_scores[0][1]


def __take_top_k(entities: [str], k: int = 10):
    """Takes the top-k entities using Mixture of Language Models approach.
    The default value is 10 as this was the default value set by Zhang and
    Balog.
    """
    # TODO: Implementation.
    pass
