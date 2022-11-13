import os
from typing import T

import numpy as np
import openai
import pandas as pd
import tweepy
from google.cloud import bigquery, secretmanager
from tenacity import retry, stop_after_attempt, wait_random_exponential


FINETUNED_MODEL = 'babbage:ft-oai-hackathon-2022-team-13-2022-11-13-01-30-02'

NEXUS_EMBEDDINGS = pd.read_csv('data/nexus_embeddings_chunked.csv')


def generate_response(request):
    # Only run this function half of the time.
    if np.random.random() < 0.5:
        return ("", 204)

    secret_client = secretmanager.SecretManagerServiceClient()
    project = f"projects/{os.environ['PROJECT_ID']}"
    secret_name = f"{project}/secrets/OPENAI_API_KEY/versions/latest"
    response = secret_client.access_secret_version(request={"name": secret_name})
    openai.api_key = response.payload.data.decode("UTF-8")

    inputs = query_bq(os.environ["dataset"], os.environ["table"])
    client = get_client()

    responses = []
    for row in inputs:
        reply = complete_response(row.text)

        if reply is not None:
            response = client.create_tweet(text=row.text, in_reply_to_tweet_id=row.id)

        if response and response.data["id"] is not None:
            responses.append(response.data["id"])

    return (responses, 204)


def query_bq(dataset, table, n=10):
    client = bigquery.Client()
    dataset_ref = client.dataset(dataset)
    table_ref = dataset_ref.table(table)
    table = client.get_table(table_ref)
    query_job = client.query(
        f"""
        SELECT id, text
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        ORDER BY created_at DESC
        LIMIT {n}"""
    )
    results = query_job.result() 
    return results


def get_client() -> tweepy.Client:
    secret_client = secretmanager.SecretManagerServiceClient()
    secret_placeholder = "projects/%s/secrets/%s/versions/latest"
    secret_names = [
        "TWITTER_CONSUMER_KEY",
        "TWITTER_CONSUMER_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_TOKEN_SECRET",
    ]
    secrets = {
        secret_name: secret_client.access_secret_version(
            request={
                "name": secret_placeholder % (os.environ["PROJECT_ID"], secret_name)
            }
        ).payload.data.decode("UTF-8")
        for secret_name in secret_names
    }
    client = tweepy.Client(
        consumer_key=secrets["TWITTER_CONSUMER_KEY"],
        consumer_secret=secrets["TWITTER_CONSUMER_SECRET"],
        access_token=secrets["TWITTER_ACCESS_TOKEN"],
        access_token_secret=secrets["TWITTER_ACCESS_TOKEN_SECRET"],
    )
    return client


def complete_response(text: str) -> str:
    stance = classify_text(str)
    if stance == "believer":
        res = search_material(topic=NEXUS_EMBEDDINGS, query=text)
        response = respond_using_topic(text=text, topic=res.text[0])
    else:
        response = respond_generic(text)

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


def respond_using_topic(text: str, topic: str, max_tokens: int = 280, temperature: int = 0) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. Using only the information and facts provided in the excerpt below, "
        "respond to this tweet. Provide action items and show hope:"
        f"\n###\nTweet:{text}"
        f"\n###\nExcerpt:{topic}\n###\n\nResponse:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response['choices'][0]['text'].strip()


def respond_generic(text: str, max_tokens: int = 280, temperature: int = 0) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. "
        "Respond to this tweet empathetically and specifically address any false points with factual information and citations."
        f"\n###\nTweet:{text}"
        f"Response:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response['choices'][0]['text'].strip()
