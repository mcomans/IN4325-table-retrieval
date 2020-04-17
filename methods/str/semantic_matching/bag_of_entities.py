# The Bag Of Entities approach is based on mapping our entities to DBpedia
# entities and finding the related entities by looking at the links
# between them.

# TODO: Add real data.
# Knowledge base entities is a list of entities in our knowledge base.
knowledge_base_entities = ["house", "garden", "door"]
# Knowledge base links is a list of related entities to the entity at the
# same index in knowledge base entities.
knowledge_base_links = [[1, 2], [0], [0]]


def semantic_map(terms: [str]) -> [[int]]:
    """Map all given terms to a list of knowledge base vectors."""
    return [__map_to_knowledge_base(term) for term in terms]


def __map_to_knowledge_base(entity: str) -> [int]:
    """Map our entity to a knowledge base vector, combined with the related
    entities."""
    try:
        entity_vec, entity_id = __entity_to_knowledge_base(entity)
        related_entities_vec = __find_related_entities(entity_id)
        return __combine_vectors(entity_vec, related_entities_vec)
    except ValueError:
        # If we could not map the entity to vector we return an empty vector.
        return __empty_vector()


def __empty_vector() -> [int]:
    """Return an empty vector the size of the knowledge base."""
    return [0] * len(knowledge_base_entities)


def __entity_to_knowledge_base(entity: str) -> ([int], int):
    """Map our original entity to our knowledge base."""
    vector = __empty_vector()
    idx = knowledge_base_entities.index(entity)
    vector[idx] = 1
    return vector, idx


def __find_related_entities(entity_id: int) -> [int]:
    """Finds the related entities in knowledge_base_links.
    :param entity_id The id of the entity in the knowledge base.
    """
    # TODO: Could implement a depth parameter that looks further than the
    #  directly related entities.
    vector = __empty_vector()
    related_entities = knowledge_base_links[entity_id]
    for entity_idx in related_entities:
        vector[entity_idx] = 1
    return vector


def __combine_vectors(a: [int], b: [int]) -> [int]:
    """Combines two feature vectors in to one."""
    return [max(x, y) for (x, y) in zip(a, b)]