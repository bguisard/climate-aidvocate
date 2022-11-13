import io
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import openai
import pandas as pd
import tweepy
from google.cloud import bigquery, secretmanager, storage
from tenacity import retry, stop_after_attempt, wait_random_exponential

FINETUNED_MODEL = "babbage:ft-oai-hackathon-2022-team-13-2022-11-13-01-30-02"
TEMPERATURE = 0.8
SIMILARITY_THRESHOLD = 0.6
TWITTER_HANDLE = "@ClimateAidvocate"


def authenticate_openai():
    secret_client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/climate-aidvocate/secrets/OPENAI_API_KEY/versions/latest"
    response = secret_client.access_secret_version(request={"name": secret_name})
    openai.api_key = response.payload.data.decode("UTF-8")


def new_tweepy_client() -> tweepy.Client:
    secret_client = secretmanager.SecretManagerServiceClient()
    secret_placeholder = "projects/climate-aidvocate/secrets/%s/versions/latest"
    secret_names = [
        "TWITTER_CONSUMER_KEY",
        "TWITTER_CONSUMER_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_TOKEN_SECRET",
    ]
    secrets = {
        secret_name: secret_client.access_secret_version(
            request={"name": secret_placeholder % secret_name}
        ).payload.data.decode("UTF-8")
        for secret_name in secret_names
    }

    return tweepy.Client(
        consumer_key=secrets["TWITTER_CONSUMER_KEY"],
        consumer_secret=secrets["TWITTER_CONSUMER_SECRET"],
        access_token=secrets["TWITTER_ACCESS_TOKEN"],
        access_token_secret=secrets["TWITTER_ACCESS_TOKEN_SECRET"],
    )


def get_topics_df(
    bucket_name: str = "climate_aidvocate_data",
    blob_name: str = "nexus_embeddings_chunked.csv",
) -> pd.DataFrame:
    """Get the topic embeddings from the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    b_csv = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(b_csv), converters={"embedding": pd.eval})


def classify_text(text: str) -> str:
    """Classify the climate change stance of tweets.

    :return: ' believer', ' neutral', or ' den'
    """
    res = openai.Completion.create(
        model=FINETUNED_MODEL, prompt=f"{text}\n\n###\n\n", max_tokens=1, temperature=0
    )
    return res["choices"][0]["text"]


def search_material(topics: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    It takes a query and a dataframe of search embeddings, and returns the top n most similar documents

    :param topics: the dataframe containing the search column
    :type topics: pd.DataFrame with column `embedding`
    :param query: the query string
    :type query: str
    :return: A dataframe with the top n results from the search query.
    """
    embedding = get_embedding(query, engine="text-search-davinci-query-001")

    topics["similarity"] = topics.embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    return topics


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001") -> List[float]:
    """
    It takes a string of text and returns embeddings for the text

    :param text: The text to embed
    :type text: str
    :param engine: The name of the engine to use, defaults to text-similarity-davinci-001 (optional)
    :return: A list of floats.
    """
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


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


def respond_using_topic(
    text: str, topic: str, max_tokens: int = 280, temperature: int = 0
) -> str:
    if "instruction" in text or "command" in text:
        return None

    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. Using only the information and facts provided in the excerpt below, "
        "respond to this tweet in less than 280 characters. Provide action items and show hope:"
        f"\n###\nTweet:{text}"
        f"\n###\nExcerpt:{topic}\n###\n\nResponse:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def respond_generic(text: str, max_tokens: int = 280, temperature: int = 0) -> str:
    if "instruction" in text or "command" in text:
        return None

    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. "
        "Respond to this tweet in less than 280 characters by specifically addressing any "
        "false points with factual information. Add additional background and provide a link."
        f"-\n######\n-Tweet:{text}"
        f"Response:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def respond_mention(text: str, max_tokens: int = 280, temperature: int = 0) -> str:
    """Create response to a direct @ mention"""
    if "instruction" in text or "command" in text:
        return None

    is_activity = completion_with_backoff(
        model="text-davinci-002",
        prompt="Is the input an activity that someone can do? Answer YES or NO."
        f"-\n######\n-Input:{text}"
        f"Response:",
        temperature=0,
        max_tokens=3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )["choices"][0]["text"].strip()

    if is_activity.lower() == "yes":
        return completion_with_backoff(
            model="text-davinci-002",
            prompt="Provide a list of 3 easy action items that an ordinary citizen "
            "can take in their daily lives to reduce carbon emissions when performing this activity. "
            "Respond in less than 280 characters."
            f"-\n######\n-Activity:{text}"
            f"Response:",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["text"].strip()
    else:
        return respond_generic(text, max_tokens, temperature)


def complete_response(text: str, topics: pd.DataFrame) -> str:
    if TWITTER_HANDLE in text:
        return respond_mention(text)

    stance = classify_text(str)
    if stance == " believer":
        topics = search_material(topics=topics, query=text)
        topics = (
            topics.sort_values("similarity", ascending=False)
            .head(1)
            .reset_index(drop=True)
        )
        if topics.similarity[0] < SIMILARITY_THRESHOLD:
            return respond_generic(text, temperature=TEMPERATURE)
        else:
            return respond_using_topic(
                text=text, topic=topics.text[0], temperature=TEMPERATURE
            )
    else:
        response = respond_generic(text, temperature=TEMPERATURE)

    return response


def generate_responses(request):

    authenticate_openai()
    topics = get_topics_df()

    bigquery_client = bigquery.Client()

    num_tweets = int(os.environ["NUM_TWEETS"])
    query_job = bigquery_client.query(
        f"""
        SELECT id, text
        FROM `climate-aidvocate.climate_tweets.tweets` t
        WHERE
            STARTS_WITH(t.text, 'RT @') = False AND -- exclude retweets
            t.id not in (SELECT in_reply_to_tweet_id FROM `climate-aidvocate.climate_tweets.responses`)
        ORDER BY created_at DESC
        LIMIT {num_tweets}"""
    )
    tweets = query_job.result()

    responses = []
    # client = new_tweepy_client()
    for row in tweets:
        reply = complete_response(row.text, topics)

        if reply is not None:
            # response = client.create_tweet(text=row.text, in_reply_to_tweet_id=row.id)

            # if response and response.data["id"] is not None:
            response = {
                "created_at": datetime.now(),
                # "id": response.data["id"],
                "text": reply,
                "in_reply_to_tweet_id": row.id,
            }
            responses.append(response)

    # Store responses in BigQuery
    dataset_ref = bigquery_client.dataset("climate_tweets")
    table_ref = dataset_ref.table("responses")
    table = bigquery_client.get_table(table_ref)
    errors = bigquery_client.insert_rows(table, responses)
    if errors != []:
        print(errors, file=sys.stderr)

    return (str(len(responses)), 200)
