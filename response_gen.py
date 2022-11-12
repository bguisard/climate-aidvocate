from typing import T

import numpy as np
import openai
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

FINETUNED_MODEL = 'ada:ft-oai-hackathon-2022-team-13-2022-11-12-20-55-09'

NEXUS_EMBEDDINGS = pd.read_csv('data/nexus_embeddings_chunked.csv')



def respond(text: str) -> str:
    response = None
    stance = classify_text(str)
    if stance == "denier":
        pass
    elif stance == "believer":
        res = search_material(topic=NEXUS_EMBEDDINGS, query=text)
        response = respond_using_topic(text=text, topic=res.text[0])

    return response


def classify_text(text: str) -> str:
    res = openai.Completion.create(model=FINETUNED_MODEL, prompt=f"{text}\n\n###\n\n", max_tokens=1, temperature=0)
    return res['choices'][0]['text']


def search_material(topics: pd.DataFrame, query: str, n=1) -> pd.DataFrame:
    """
    It takes a query and a dataframe of search embeddings, and returns the top n most similar documents

    :param topics: the dataframe containing the search column
    :type topics: pd.DataFrame with column `embedding`
    :param query: the query string
    :type query: str
    :param n: the number of results to return, defaults to 3 (optional)
    :return: A dataframe with the top n results from the search query.
    """
    embedding = get_embedding(query, engine="text-search-davinci-query-001")

    topics["similarities"] = topics.embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    return topics.sort_values("similarities", ascending=False).head(n).reset_index(drop=True)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(
    text: str, engine="text-similarity-davinci-001"
) -> T.List[float]:
    """
    It takes a string of text and returns embeddings for the text

    :param text: The text to embed
    :type text: str
    :param engine: The name of the engine to use, defaults to text-similarity-davinci-001 (optional)
    :return: A list of floats.
    """
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0][
        "embedding"
    ]


def cosine_similarity(a, b):
    """
    It takes two vectors, a and b, and returns the cosine of the angle between them

    :param a: the first vector
    :param b: the number of bits to use for the hash
    :return: The cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def respond_using_topic(text: str, topic: str, max_tokens: int = 512, temperature: int = 0) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. Using only the information and facts provided in the excerpt below, "
        "respond to this message in less than 280 characters. Provide action items and show hope:"
        f"\n###\nMessage:{text}"
        f"\n###\nExcerpt:{topic}\n###\n\nResponse:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response['choices'][0]['text'].strip()
