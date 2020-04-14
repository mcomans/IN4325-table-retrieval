from methods.str import extraction, semantic_matching
from methods.types import Table, Query

# Semantic Table Retrieval is made up of three major steps:
# - Extraction of table and query data to terms (word or entity based)
# - Mapping of terms to semantic space
# - Similarity calculation


def run_str(e_method: extraction.ExtractionMethod,
            sm_method: semantic_matching.SemanticSpace):
    """Run the STR method given a set of parameters.
    :param e_method The type of extraction method used.
    :param sm_method The type of semantic mapping used.
    """
    table = Table()  # Placeholder
    query = Query()  # Placeholder

    # Extraction step
    t_extracted_words = \
        extraction.extract(table, query, e_method)
    q_extracted_words = \
        extraction.extract(table, query, e_method)

    # Mapping Step
    t_semantic_vectors = semantic_matching.to_semantic_space(
        t_extracted_words, sm_method)
    q_semantic_vectors = semantic_matching.to_semantic_space(
        q_extracted_words, sm_method)

    # TODO: Similarity Calculation
