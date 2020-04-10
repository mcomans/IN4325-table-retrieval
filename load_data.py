from methods.types import Table, Query
import json
import re


def read_tables(filename="example.json") -> [Table]:
    """Reads a json file in the data directory and outputs the tables parsed
    from it."""
    with open('data/tables/{filename}'.format(filename=filename), 'r') as file:
        data = json.loads(file.read())
        return [Table(id=table, **data[table]) for table in data]


def read_queries(filename="example.txt") -> [Query]:
    """Read a file in the data directory and output the queries parsed from
    it."""
    queries = []
    with open('data/queries/{filename}'.format(filename=filename), 'r') as file:
        return [__parse_query(q) for q in file]


def __parse_query(input: str) -> Query:
    """Helper function to parse queries."""
    matches = re.match(r"(\d+) (.*)$", input)
    return Query(id=int(matches[1]), query=matches[2])
