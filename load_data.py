from methods.types import Table, Query
import json
import re


def read_tables(filename="data/tables/example.json") -> [Table]:
    """Reads a json file in the data directory and outputs the tables parsed
    from it."""
    with open(filename, 'r') as file:
        data = json.loads(file.read())
        return [Table(id=table, **data[table]) for table in data]


def read_queries(filename="data/queries/example.txt") -> [Query]:
    """Read a file in the data directory and output the queries parsed from
    it."""
    with open(filename, 'r') as file:
        return [__parse_query(q) for q in file]


def __parse_query(input: str) -> Query:
    """Helper function to parse queries."""
    matches = re.match(r"(\d+) (.*)$", input)
    return Query(id=int(matches[1]), query=matches[2])


def get_table(table_dir: str, table_id: str) -> Table or None:
    split_table_id = table_id.split("-")
    table_file = f"re_tables-{split_table_id[1]}.json"
    tables = read_tables(f"{table_dir}/{table_file}")
    tables_filtered = [x for x in tables if x.id == table_id]
    if len(tables_filtered) < 1:
        print(f"Could not find table {table_id} in file {table_file}")
        return None

    return tables_filtered[0]