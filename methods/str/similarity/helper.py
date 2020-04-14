from numpy import dot
from numpy.linalg import norm


def cosine_sim(a: [int], b: [int]) -> float:
    """Calculates the cosine similarity given two vectors."""
    return dot(a, b) / (norm(a) * norm(b))
