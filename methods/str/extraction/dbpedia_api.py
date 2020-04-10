import requests

cache = {}


def extract_entities(input: str) -> [str]:
    cached = __get_cached(input)
    if cached is None:
        return __make_request(input)
    else:
        return cached


def __make_request(input: str) -> [str]:
    response = requests.get("https://api.dbpedia-spotlight.org/en/annotate",
                            headers={"accept": "application/json"},
                            params={"text": input})
    response_data = response.json()
    if hasattr(response_data, "Resources"):
        result = [resource["@surfaceForm"]
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
