import requests


url = "http://localhost:2222"
confidence_score = 0.1
cache = {}


def extract_entities(input: str) -> [str]:
    cached = __get_cached(input)
    if cached is None:
        return __make_request(input)
    else:
        return cached


def __make_request(input: str) -> [str]:
    response = requests.get(f"{url}/rest/annotate",
                            headers={"accept": "application/json"},
                            params={"text": input, "confidence": confidence_score})
    if response.status_code != 200:
        result = []
    else:
        response_data = response.json()
        if "Resources" in response_data:
            result = [__resource_uri_to_entity(resource["@URI"])
                      for resource in response.json()["Resources"]]
        else:
            result = []
    __add_to_cache(input, result)
    return result


def __get_cached(input: str) -> [str]:
    if hasattr(cache, input):
        return cache[input]


def __add_to_cache(input: str, result: [str]):
    cache[input] = result


def __resource_uri_to_entity(uri: str) -> str:
    """Return the actual entity which is described in the last part of the
    uri as for example with http://dbpedia.org/resource/World_Wide_Web"""
    sections = uri.split("/")
    return sections[len(sections) - 1]
