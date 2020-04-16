import argparse
import pandas as pd
from load_data import read_queries, get_table, read_qrels
from methods.feature_extractor.utils import total_term_frequency, total_idf

parser = argparse.ArgumentParser(description="Extracts features from query-table combinations")
parser.add_argument("-q", "--queries", required=True, help="Query file")
parser.add_argument("-r", "--rel", required=True, help="Relevance file in qrel format")
parser.add_argument("-t", "--tables", default="data/tables", help="Folder with tables in json format")
parser.add_argument("-s", "--str", default="data/STR_features.csv", help="STR features file")

args = parser.parse_args()

queries = {q.id: q.query for q in read_queries(args.queries)}
rels = read_qrels(args.rel)
str_features = pd.read_csv(args.str)

tables = [table for table in [get_table(args.tables, rel[1]) for rel in rels] if table]
page_titles = [table.page_title.lower().split() for table in tables]
section_titles = [table.second_title.lower().split() for table in tables]
table_captions = [table.caption.lower().split() for table in tables]
table_bodies = [table.body_tokens() for table in tables]
table_headers = [table.header_tokens() for table in tables]


def __rel_to_feature_df(rel):
    query_id = rel[0]
    query = queries[int(query_id)]
    table_id = rel[1]
    relevance = rel[2]

    filtered_tables = [table for table in tables if table.id == table_id]
    if len(filtered_tables) < 1:
        return None

    table = filtered_tables[0]
    query_terms = query.split()

    q_in_pg_title = sum(1 for term in query_terms if term in table.page_title.lower().split()) / len(query_terms)
    q_in_table_title = sum(1 for term in query_terms if term in table.caption.lower().split()) / len(query_terms)

    features = {
        "query_id": int(query_id),
        "query": query,
        "table_id": table_id,
        "QLEN": len(query_terms),
        "IDF_pgTitle": total_idf(query_terms, page_titles),
        "IDF_secTitle": total_idf(query_terms, section_titles),
        "IDF_caption": total_idf(query_terms, table_captions),
        "IDF_header": total_idf(query_terms, table_headers),
        "IDF_body": total_idf(query_terms, table_bodies),
        "null": table.empty_cell_count(),
        "row": table.num_data_rows,
        "col": table.num_cols,
        "leftColHits": total_term_frequency(query_terms, table.column_tokens(0)),
        "secColHits": total_term_frequency(query_terms, table.column_tokens(1)),
        "bodyHits": total_term_frequency(query_terms, table.body_tokens()),
        "qInPgTitle": q_in_pg_title,
        "qInTableTitle": q_in_table_title,
        "rel": relevance
    }

    return pd.DataFrame(features, index=[0])


baseline_features = pd.concat([__rel_to_feature_df(rel) for rel in rels])
baseline_features.to_csv("data/baseline_features.csv", header=True, index=False)

all_features = pd.merge(baseline_features, str_features, on=['query_id', 'table_id'])
all_features.to_csv("data/all_features.csv", header=True, index=False)
