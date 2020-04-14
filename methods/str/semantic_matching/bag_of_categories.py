from . import dbpedia_sparql

# The Bag Of Categories approach is based on mapping our entities to
# Wikipedia categories. As it is not explained in the paper by Zhang and
# Balog we have written our own implementation.

# This approach is not versionable as it relies on sparql queries made to
# dbpedia and DPpedia is continously updated.


def semantic_map(terms: [str]):
    return [__entity_to_category(term) for term in terms]


def __entity_to_category(entity: str) -> [int]:
    """Maps the entity to a category vector with a 1 for the matching
    category if one is found."""
    vec = __empty_vector()
    for cat_id in dbpedia_sparql.subjects_for_entity(entity):
        vec[cat_id] = 1
    return vec


def __empty_vector() -> [int]:
    return [0] * (dbpedia_sparql.MAX_SUBJECT_ID + 1)