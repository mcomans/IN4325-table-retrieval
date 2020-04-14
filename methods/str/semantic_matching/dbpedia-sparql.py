from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# The max subject id is retrieved from the following sparql query:
# SELECT ?subId WHERE {
#     ?uri dct:subject ?subject.
#     ?subject dbo:wikiPageID ?subId
# } ORDER BY DESC(?subId) LIMIT 1
MAX_SUBJECT_ID = 52049076



def subjects_for_entity(entity: str) -> [int]:
    """
    Returns the dbo:wikiPageID for the subjects related to the entity.

    :param entity A dbpedia entity url like World_Wide_Web. This comes from
    the original resource url http://dbpedia.org/resource/World_Wide_Web.
    """
    sparql.setQuery(f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?subId ?subject WHERE {{
        ?uri dct:subject ?subject.
        ?subject dbo:wikiPageID ?subId
        FILTER(?uri = <http://dbpedia.org/resource/{entity}>)
    }} LIMIT 50
    """)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    return [int(b['subId']['value']) for b in result['results']['bindings']]


print(subjects_for_entity("http://dbpedia.org/resource/World_Wide_Web"))
