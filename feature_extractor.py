import argparse
from load_data import read_queries, get_table, read_qrels
from methods.feature_extractor.utils import total_term_frequency, total_idf

parser = argparse.ArgumentParser(description="Extracts features from query-table combinations")
parser.add_argument("-q", "--queries", required=True, help="Query file")
parser.add_argument("-r", "--rel", required=True, help="Relevance file in qrel format")
parser.add_argument("-t", "--tables", default="data/tables", help="Folder with tables in json format")
parser.add_argument("-o", "--output", default="data/features.csv", help="Feature output file")

args = parser.parse_args()

queries = {q.id: q.query for q in read_queries(args.queries)}

rels = read_qrels(args.rel)

tables = [table for table in [get_table(args.tables, rel[1]) for rel in rels] if table]
page_titles = [table.page_title.lower().split() for table in tables]
section_titles = [table.second_title.lower().split() for table in tables]
table_captions = [table.caption.lower().split() for table in tables]
table_bodies = [table.body_tokens() for table in tables]
table_headers = [table.header_tokens() for table in tables]


with open(args.output, "w") as out:
    out.write("query_id,"
              "query,"
              "table_id,"
              "QLEN,"
              "IDF_pgTitle,"
              "IDF_secTitle,"
              "IDF_caption,"
              "IDF_header,"
              "IDF_body,"
              "null,"
              "row,"
              "col,"
              "leftColHits,"
              "secColHits,"
              "bodyHits,"
              "qInPgTitle,"
              "qInTableTitle,"
              "rel\n")

    for rel in rels:
        query_id = rel[0]
        query = queries[int(query_id)]
        table_id = rel[1]
        relevance = rel[2]

        filtered_tables = [table for table in tables if table.id == table_id]
        if len(filtered_tables) < 1:
            continue

        table = filtered_tables[0]
        query_terms = query.split()

        q_in_pg_title = sum(1 for term in query_terms if term in table.page_title.lower().split()) / len(query_terms)
        q_in_table_title = sum(1 for term in query_terms if term in table.caption.lower().split()) / len(query_terms)

        out.write(f"{query_id},"
                  f"{query},"
                  f"{table_id},"
                  f"{len(query_terms)},"
                  f"{total_idf(query_terms, page_titles)},"
                  f"{total_idf(query_terms, section_titles)},"
                  f"{total_idf(query_terms, table_captions)},"
                  f"{total_idf(query_terms, table_headers)},"
                  f"{total_idf(query_terms, table_bodies)},"
                  f"{table.empty_cell_count()},"
                  f"{table.num_data_rows},"
                  f"{table.num_cols},"
                  f"{total_term_frequency(query_terms, table.column_tokens(0))},"
                  f"{total_term_frequency(query_terms, table.column_tokens(1))},"
                  f"{total_term_frequency(query_terms, table.body_tokens())},"
                  f"{q_in_pg_title},"
                  f"{q_in_table_title},"
                  f"{relevance}")
