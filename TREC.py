from pathlib import Path
import time


def write_results(results, run_id):
    """
    Writes results to a file in the TREC results format, which can then be used in trec_eval.
    :param results: The results dataframe containing a score column and the query information data
    (query_id, query, table_id).
    :param run_id: An identifier for the run, which will be used as filename and for the STANDARD field in TREC results.
    """
    Path("results").mkdir(parents=True, exist_ok=True)
    results = results.groupby('query_id').apply(lambda x: x.sort_values(['score'], ascending=False)).reset_index(
        drop=True)
    with open(f'results/{time.strftime("%Y%m%d-%H%M%S")}_{run_id}.txt', 'w') as file:
        for index, row in results.iterrows():
            file.write(f"{row['query_id']} Q0 {row['table_id']} 1 {row['score']} {run_id}\n")
