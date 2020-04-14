import requests
import pickle
from os import path

url = "http://localhost:2222"
confidence_score = 0.1

if path.exists("data/dbpedia_api.cache"):
    with open("data/dbpedia_api.cache", "rb") as load_cache:
        cache = pickle.load(load_cache)
        print(f"Loaded {len(cache)} items from the cache for dbpedia-api.")
else:
    cache = {}


def extract_entities(input: str) -> [str]:
    cached = __get_cached(input)
    if cached is not None:
        return cached

    return __make_request(input)


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
    if str(input) in cache:
        return cache[str(input)]
    return None


def __add_to_cache(input: str, result: [int]):
    cache[str(input)] = result
    with open("data/dbpedia_api.cache", "wb") as write_cache:
        pickle.dump(cache, write_cache)


def __resource_uri_to_entity(uri: str) -> str:
    """Return the actual entity which is described in the last part of the
    uri as for example with http://dbpedia.org/resource/World_Wide_Web"""
    sections = uri.split("/")
    return sections[len(sections) - 1]
