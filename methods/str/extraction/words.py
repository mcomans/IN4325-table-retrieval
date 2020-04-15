from methods.types import Table, Query


def extract_unique_words(input) -> [str]:
    """Retrieves the unique words from the input."""
    terms = None
    if type(input) == Table:
        terms = __extract_table_terms(input)
    elif type(input) == Query:
        terms = input.query.split(" ")
    if terms is None:
        return []
    return list(set(terms))


def __extract_table_terms(table: Table) -> [str]:
    """Uses some aspects of the Table input to retrieve the terms relevant as
    defined by Zhang and Balog."""
    return " ".join(table.title + [table.caption]).split(" ")