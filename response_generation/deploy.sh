#!/bin/bash
set -eu -o pipefail

gcloud functions deploy response-generation \
  --runtime python39 \
  --region us-west1 \
  --timeout 300 \
  --trigger-http \
  --entry-point=generate_response \
  --set-env-vars PROJECT_ID=climate-aidvocate
