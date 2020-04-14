from methods.types import Table, Query
import json
import re
import pandas as pd


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


def read_features(filename='data/features.csv'):
    """Reads a features file in csv format from the data directory and builds a Pandas DataFrame from it."""
    return pd.read_csv(filename)


def read_qrels(filename='data/qrels.txt') -> [(int, str, int)]:
    """Reads a qrels file returning pairs of q_id table_id and rel_score"""
    with open(filename, "r") as rel_file:
        split_lines = [line.split('\t') for line in rel_file]
        return [(split[0], split[2], split[3]) for split in split_lines]
