from methods.str import extraction, semantic_matching, similarity
from methods.types import Table, Query
from load_data import read_queries, read_qrels, get_table
from TREC import write_results
import pandas as pd

# Semantic Table Retrieval is made up of three major steps:
# - Extraction of table and query data to terms (word or entity based)
# - Mapping of terms to semantic space
# - Similarity calculation


def run_str(e_method: extraction.ExtractionMethod,
            sm_method: semantic_matching.SemanticSpace,
            sim_method: similarity.SimilarityMethod):
    """Run the STR method given a set of parameters.
    :param e_method The type of extraction method used.
    :param sm_method The type of semantic mapping used.
    :param sim_method The type of similarity method used.
    """
    qrels = read_qrels('data/qrels.txt')
    queries = read_queries('data/queries/queries.txt')
    results = []
    for (q_id, t_id, rel_score) in qrels:
        query = queries[int(q_id) - 1]
        table = get_table('data/tables/', t_id)
        score = run_for_query_table_pair(query, table, e_method, sm_method, sim_method)
        # Given that we are using late fusion our score will be a tuple itself.
        results.append({
            "query_id": q_id,
            "table_id": t_id,
            "score": score
        })

    # Write results in TREC format.
    if sim_method == similarity.SimilarityMethod.EARLY_FUSION:
        write_results(pd.DataFrame(results),
                      f"STR_{e_method}_{sm_method}_{sim_method}")
    else:
        def __use_score(r: dict, id: int):
            result = r
            result["score"] = r["score"][id]
            return result
        results_max = [__use_score(result, 0) for result in results]
        write_results(pd.DataFrame(results_max),
                      f"STR_{e_method}_{sm_method}_{sim_method}_MAX")
        results_sum = [__use_score(result, 1) for result in results]
        write_results(pd.DataFrame(results_sum),
                      f"STR_{e_method}_{sm_method}_{sim_method}_SUM")
        results_avg = [__use_score(result, 2) for result in results]
        write_results(pd.DataFrame(results_avg),
                      f"STR_{e_method}_{sm_method}_{sim_method}_AVG")


def run_for_query_table_pair(query: Query, table: Table,
                             e_method: extraction.ExtractionMethod,
                             sm_method: semantic_matching.SemanticSpace,
                             sim_method: similarity.SimilarityMethod) -> float:
    # Extraction step
    t_extracted_words, q_extracted_words = \
        extraction.extract(table, query, e_method)

    # Mapping Step
    t_semantic_vectors = semantic_matching.to_semantic_space(
        t_extracted_words, sm_method)
    q_semantic_vectors = semantic_matching.to_semantic_space(
        q_extracted_words, sm_method)

    # Similarity Calculation
    return similarity.calculate_sim(q_semantic_vectors, t_semantic_vectors, sim_method)