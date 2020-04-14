from .helper import cosine_sim

# Early fusion calculates the similarity between the vectors from the query
# by first representing them as a single vector. This similarity score is
# then determined by cosine similarity.
# The different vectors are combined using the centroid vector. An exception
# occurs when we are using the word term (check extraction) we should employ
# a weighting using standard TF-IDF.


def calculate_sim(query_vectors: [[int]],
                  table_vectors: [[int]]) -> float:
    # TODO: Use specific TF-IDF weighting in case of word terms.
    # However it is not clear compared to which documents this should be done
    # according to Zhang and Bolag.
    query_centroid = __calculate_centroid(query_vectors)
    table_centroid = __calculate_centroid(table_vectors)
    assert len(query_centroid) == len(table_centroid)
    if len(query_centroid) == 1:
        return abs(query_centroid - table_centroid)
    return cosine_sim(query_centroid, table_centroid)


def __calculate_centroid(vectors: [[int]]) -> [int]:
    """Combine all vectors in to a centroid."""
    assert len(vectors) > 0
    vector_length = len(vectors[0])
    result_vec = [0] * vector_length
    for i in range(vector_length):
        result_vec = sum([v[i] for v in vectors]) / vector_length
    return result_vec
