from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# The max subject id is retrieved from the following sparql query:
# SELECT ?subId WHERE {
#     ?uri dct:subject ?subject.
#     ?subject dbo:wikiPageID ?subId
# } ORDER BY DESC(?subId) LIMIT 1
MAX_SUBJECT_ID = 52049076

cache = {}


def subjects_for_entity(entity: str) -> [int]:
    """
    Returns the dbo:wikiPageID for the subjects related to the entity.

    :param entity A dbpedia entity url like World_Wide_Web. This comes from
    the original resource url http://dbpedia.org/resource/World_Wide_Web.
    """
    cached = __get_cached(entity)
    if cached:
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
    if hasattr(cache, input):
        return cache[input]
    return None


def __add_to_cache(input: str, result: [int]):
    cache[input] = result
