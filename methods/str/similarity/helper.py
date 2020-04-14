from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(a: [float], b: [float]) -> float:
    """Calculates the cosine similarity given two vectors."""
    return cosine_similarity([a], [b])[0][0]
