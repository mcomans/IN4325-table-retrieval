from enum import Enum
from . import bag_of_entities, bag_of_categories, word_embeddings, graph_embeddings


class SemanticSpace(Enum):
    BAG_OF_ENTITIES = "BAG_OF_ENTITIES"
    BAG_OF_CATEGORIES = "BAG_OF_CATEGORIES"
    WORD_EMBEDDINGS = "WORD_EMBEDDINGS"
    GRAPH_EMBEDDINGS = "GRAPH_EMBEDDINGS"


def to_semantic_space(terms: [str], target_space: SemanticSpace):
    """Move our terms (either entities or words) in to a semantic space."""
    if target_space == SemanticSpace.BAG_OF_CATEGORIES:
        return bag_of_entities.semantic_map(terms)
    elif target_space == SemanticSpace.BAG_OF_CATEGORIES:
        return bag_of_categories.semantic_map(terms)
    elif target_space == SemanticSpace.WORD_EMBEDDINGS:
        return word_embeddings.semantic_map(terms)
    elif target_space == SemanticSpace.GRAPH_EMBEDDINGS:
        return graph_embeddings.semantic_map(terms)
