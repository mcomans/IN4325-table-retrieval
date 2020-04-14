from enum import Enum
from . import early_fusion, late_fusion


class SimilarityMethod(Enum):
    EARLY_FUSION = "EARLY_FUSION"
    LATE_FUSION = "LATE_FUSION"


def calculate_sim(query_vectors: [[int]],
                  table_vectors: [[int]],
                  method: SimilarityMethod):
    """Calculate the similarity based on the given method."""
    if method == SimilarityMethod.EARLY_FUSION:
        return early_fusion.calculate_sim(query_vectors, table_vectors)
    elif method == SimilarityMethod.LATE_FUSION:
        lf_max, lf_sum, lf_avg = late_fusion.calculate_sim(query_vectors,
                                                           table_vectors)
        return lf_max, lf_sum, lf_avg
