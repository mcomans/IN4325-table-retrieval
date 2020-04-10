from methods.str import extraction
from methods.str.types import Table, Query

# Semantic Table Retrieval is made up of three major steps:
# - Extraction of table and query data to terms (word or entity based)
# - Mapping of terms to semantic space
# - Similarity calculation


def run_str(e_method: extraction.ExtractionMethod):
    """Run the STR method given a set of parameters.
    :param e_method The type of extraction method used.
    """
    table = Table()  # Placeholder
    query = Query()  # Placeholder

    # Extraction step
    t_extracted_words = \
        extraction.extract(table, query, e_method)
    q_extracted_words = \
        extraction.extract(table, query, e_method)

    # TODO: Mapping Step

    # TODO: Similarity Calculation
