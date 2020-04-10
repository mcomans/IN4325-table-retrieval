import argparse
from load_data import read_queries, read_tables

parser = argparse.ArgumentParser(description="Extracts features from query-table combinations")
parser.add_argument("-q", "--queries", required=True, help="Query file")
parser.add_argument("-r", "--rel", required=True, help="Relevance file in qrel format")
parser.add_argument("-t", "--tables", default="data/tables", help="Folder with tables in json format")
parser.add_argument("-o", "--output", default="data/features.csv", help="Feature output file")

args = parser.parse_args()

queries = {q.id: q.query for q in read_queries(args.queries)}

with open(args.rel, "r") as rel_file:
    split_lines = [line.split('\t') for line in rel_file]
    rels = [(split[0], split[2], split[3]) for split in split_lines]

with open(args.output, "w") as out:
    out.write("query_id,query,table_id,qlen,row,col,rel\n")

    for rel in rels:
        query_id = rel[0]
        query = queries[int(query_id)]
        table_id = rel[1]
        relevance = rel[2]

        split_table_id = table_id.split("-")
        table_file = f"re_tables-{split_table_id[1]}.json"
        tables = read_tables(f"{args.tables}/{table_file}")
        tables_filtered = [x for x in tables if x.id == table_id]
        if len(tables_filtered) < 1:
            print(f"Could not find table {table_id} in file {table_file}")
            continue

        table = tables_filtered[0]

        out.write(f"{query_id},{query},{table_id},{len(query.split())},{table.num_data_rows},{table.num_cols},{relevance}")
