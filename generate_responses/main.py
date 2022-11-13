import io
import os
import sys
from datetime import datetime
from typing import List

import openai
import pandas as pd
import tweepy
from google.cloud import bigquery, secretmanager, storage

from response_gen import (
    respond_generic,
    respond_mention,
    respond_using_topic,
    search_material,
    split_responses
)

FINETUNED_MODEL = "babbage:ft-oai-hackathon-2022-team-13-2022-11-13-01-30-02"
TEMPERATURE = 0.8
SIMILARITY_THRESHOLD = 0.6
TWITTER_HANDLE = "@ClimateAidvocate"
TOKEN_THRESHOLD = 550


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


def complete_response(text: str, topics: pd.DataFrame) -> List:
    if TWITTER_HANDLE in text:
        return [(respond_mention(text))]

    stance = classify_text(str)
    if stance == " believer":
        topics = search_material(topics=topics, query=text)
        topics = (
            topics.sort_values("similarity", ascending=False)
            .head(1)
            .reset_index(drop=True)
        )
        if topics.similarity[0] < SIMILARITY_THRESHOLD:
            return [respond_generic(text, temperature=TEMPERATURE)]
        else:
            return [respond_using_topic(text=text, topic=topics.text[0], temperature=TEMPERATURE)]
    else:
        response = respond_generic(text, temperature=TEMPERATURE, max_tokens=TOKEN_THRESHOLD)
        return split_responses(response)


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
            for part in reply:
                response = {
                    "created_at": datetime.now(),
                    # "id": response.data["id"],
                    "text": part,
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

    return (len(responses), 200)
