#!/bin/bash
set -eu -o pipefail

gcloud functions deploy generate-responses \
  --runtime python39 \
  --region us-west1 \
  --timeout 300 \
  --trigger-http \
  --entry-point=generate_responses \
  --set-env-vars NUM_TWEETS=20
