from SPARQLWrapper import SPARQLWrapper, JSON
from os import path
import pickle

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# The max subject id is retrieved from the following sparql query:
# SELECT ?subId WHERE {
#     ?uri dct:subject ?subject.
#     ?subject dbo:wikiPageID ?subId
# } ORDER BY DESC(?subId) LIMIT 1
MAX_SUBJECT_ID = 52049076

if path.exists("data/dbpedia_sparql.cache"):
    with open("data/dbpedia_sparql.cache", "rb") as load_cache:
        cache = pickle.load(load_cache)
        print(f"Loaded {len(cache)} items from the cache for dbpedia-sparql")
else:
    cache = {}


def subjects_for_entity(entity: str) -> [int]:
    """
    Returns the dbo:wikiPageID for the subjects related to the entity.

    :param entity A dbpedia entity url like World_Wide_Web. This comes from
    the original resource url http://dbpedia.org/resource/World_Wide_Web.
    """
    cached = __get_cached(entity)
    if cached is not None:
        return cached

    sparql.setQuery(f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?subId ?subject WHERE {{
        ?uri dct:subject ?subject.
        ?subject dbo:wikiPageID ?subId
        FILTER(?uri = <http://dbpedia.org/resource/{entity}>)
    }} LIMIT 50
    """)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()
    category_ids = [int(b['subId']['value'])
                    for b in response['results']['bindings']]
    __add_to_cache(entity, category_ids)
    return category_ids


def __get_cached(input: str) -> [int]:
    if str(input) in cache:
        return cache[str(input)]
    return None


def __add_to_cache(input: str, result: [int]):
    cache[str(input)] = result
    with open("data/dbpedia_sparql.cache", "wb") as write_cache:
        pickle.dump(cache, write_cache)
