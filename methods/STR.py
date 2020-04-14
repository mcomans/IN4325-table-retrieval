from methods.str import extraction, semantic_matching, similarity
from methods.types import Table, Query
from load_data import read_queries, read_qrels, get_table

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
    for (q_id, t_id, rel_score) in qrels:
        query = queries[int(q_id)]
        table = get_table('data/tables/', t_id)
        score = run_for_query_table_pair(query, table, e_method, sm_method, sim_method)
        print(f"[q{q_id}-{t_id}][r-{rel_score}] -> {score}")


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