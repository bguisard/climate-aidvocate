import base64
import json
import os
import sys
from datetime import datetime

from google.cloud import bigquery


def write_tweets_to_bq(dataset, table, document):
    bigquery_client = bigquery.Client()
    dataset_ref = bigquery_client.dataset(dataset)
    table_ref = dataset_ref.table(table)
    table = bigquery_client.get_table(table_ref)

    row = {
        "created_at": datetime.now(),
        "id": document["data"]["id"],
        "text": document["data"]["text"],
    }
    errors = bigquery_client.insert_rows(table, [row])
    if errors != []:
        print(errors, file=sys.stderr)


def consume_tweet(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event["data"]).decode("utf-8")

    write_tweets_to_bq(
        os.environ["dataset"], os.environ["table"], json.loads(pubsub_message)
    )
