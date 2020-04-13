from enum import Enum
from . import early_fusion


class SimilarityMethod(Enum):
    EARLY_FUSION = "EARLY_FUSION"
    LATE_FUSION_SUM = "LATE_FUSION_SUM"
    LATE_FUSION_AVG = "LATE_FUSION_AVG"
    LATE_FUSION_MAX = "LATE_FUSION_MAX"


def calculate_sim(query_vectors: [[int]],
                  table_vectors: [[int]],
                  method: SimilarityMethod) -> float:
    """Calculate the similarity based on the given method."""
    if method == SimilarityMethod.EARLY_FUSION:
        return early_fusion.calculate_sim(query_vectors, table_vectors)