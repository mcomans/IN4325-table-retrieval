from methods.types import Table
import json


def read_tables(filename="example.json") -> [Table]:
    """Reads a json file in the data directory and outputs the tables parsed
    from it."""
    with open('data/tables/{filename}'.format(filename=filename), 'r') as file:
        data = json.loads(file.read())
        return [Table(id=table, **data[table]) for table in data]
