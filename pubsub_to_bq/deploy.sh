#!/bin/bash
set -eu -o pipefail

gcloud functions deploy pubsub-to-bq \
  --runtime python39 \
  --region us-west1 \
  --timeout 120 \
  --trigger-event=google.pubsub.topic.publish \
  --trigger-resource=raw-tweets \
  --entry-point=consume_tweet \
  --set-env-vars dataset=climate_tweets,table=tweets
