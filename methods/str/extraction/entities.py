from . import dbpedia_api
from methods.types import Table, Query


def extract_entities(input) -> [str]:
    """Extract named entities from the given terms."""
    terms = None
    if type(input) == Table:
        terms = __extract_table_terms(input)
    elif type(input) == Query:
        terms = input.query.split(" ")
    else:
        terms = input
    assert terms is not None
    return dbpedia_api.extract_entities(" ".join(terms))


def __extract_table_terms(table: Table) -> [str]:
    """Uses some aspects of the Table input to retrieve the terms relevant as
    defined by Zhang and Balog. For entity extraction this is the core
    column, page title and table caption."""
    aspects = __find_core_column(table) + [table.page_title, table.caption]
    return " ".join(aspects).split(" ")


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
