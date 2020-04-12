# The Bag Of Categories approach is based on mapping our entities to
# Wikipedia categories. As it is not explained in the paper by Zhang and
# Balog we have written our own implementation.


def semantic_map(terms: [str]):
    return [__entity_to_category(term) for term in terms]


def __entity_to_category(entity: str) -> [int]:
    """Maps the entity to a category vector with a 1 for the matching
    category if one is found."""
    pass