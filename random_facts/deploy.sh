#!/bin/bash
set -eu -o pipefail

gcloud functions deploy random-facts \
  --runtime python39 \
  --region us-west1 \
  --timeout 300 \
  --trigger-http \
  --entry-point=generate_fun_fact
