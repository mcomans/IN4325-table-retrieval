# IN4325-table-retrieval
Table retrieval using the semantic space.

## Reproduced paper
Shuo Zhang and Krisztian Balog. 2018. *Ad Hoc Table Retrieval using Semantic Similarity*. In Proceedings of the 2018 World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 1553–1562.

This paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3178876.3186067).

## Datasets

We use similar data as in the original paper which is described [here](https://github.com/iai-group/www2018-table/). 
For the application to run we expect the tables to be located in the the `data/tables/` folder and the queries in the `data/queries/` folder.
The relation between the tables and queries (named `qrels` in the original repo) should be placed in the `data/` folder.

## Running Locally

This section explains how to set up and run the code locally.

### Prerequisites

In our tests we used Python 3 (`3.7.7`) which is what we recommend you to
 install.
It is optional to create a virtualenv with this python version as described
 in step 2, this requires you to have virtualenv (`16.6.2`) installed on your
 system too.
 
### Installation steps

1. Clone the repository to a place of your liking.
2. `Optional` Create a virtual environment using
[virtualenv](https://virtualenv.pypa.io/en/stable/):
    ```bash
    # Create the virtualenv.
    python3 -m virtualenv --python=/location/to/python3.7.7/local/bin/python .venv

    # Activate the virtualenv.
    source .venv/bin/activate
    ```
3. Install the required dependencies specified in `requirements.txt`:
    ```bash
    python -m pip install -r requirements.txt
    ```
4. Install spaCy requirements:
    ```bash
   python -m spacy download en_core_web_sm
   ```
5. `Optional but recommended` Setup a docker image running the [spotlight api](https://github.com/dbpedia-spotlight/spotlight-docker).
    ```bash
    docker run -itd --restart unless-stopped -p 2222:80 dbpedia/spotlight-english spotlight.sh
    ```
    It is expected to run at `localhost:2222` as specified in `/methods/str/dbpedia_api.py`. If you run the image on another ip or port please adjust this in the `dbpedia_api` file.

### Running steps

The application can be run in a few different modes to run part or all the experiments mentioned in our paper, you can look in `experiments.py` or use the command `python experiments.py -h` to see all available modes.
