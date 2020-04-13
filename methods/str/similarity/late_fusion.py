from .helper import cosine_sim


# Late Fusion works by calculating the similarity in a pairwise fashion
# between the query and table vectors. These are then aggregated using
# different functions: max, sum, avg.


def calculate_sim(query_vectors: [[int]],
                  table_vectors: [[int]]) -> (float, float, float):
    similarities = [cosine_sim(a, b)
                    for a in query_vectors
                    for b in table_vectors]
    # To avoid recomputation of the similarities (which is quite expensive)
    # we decide to calculate all the different variants of late fusion in one
    # go.
    similarities_sum = sum(similarities)
    return max(similarities), \
           similarities_sum, \
           similarities_sum / len(similarities)
