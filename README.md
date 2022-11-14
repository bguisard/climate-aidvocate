# The Climate AIdvocate

Serving code for the [@ClimateAIdvocat]. The climate AIdvocate is a GPT-3 powered
twitter bot that flags false or misleading tweets about climate change, and
responds with useful resources.

## Real-time tweet pipeline

1. Twitter to Pub/Sub

    An application deployed with the App Engine that monitors tweets in real-time
    and pushes them to a Pub/Sub topic. Every tweet that matches any of the
    `StreamRule`s will be picked up by the `PubSubStreamer`.

2. Pub/Sub to BigQuery

    Tweets published to the topic then trigger a Cloud Function that parses the
    tweet and adds them to BigQuery.


Based on:
- [Real-time data analysis using Kubernetes, PubSub, and BigQuery]
- [Real time tweets pipeline using GCP]
- [Streaming data from Twitter to GCP]


[@ClimateAIdvocat]: https://twitter.com/ClimateAIdvocat

[Real-time data analysis using Kubernetes, PubSub, and BigQuery]:https://github.com/GoogleCloudPlatform/kubernetes-bigquery-python/blob/master/pubsub/README.md

[Real time tweets pipeline using GCP]: https://github.com/polleyg/gcp-tweets-streaming-pipeline

[Streaming data from Twitter to GCP]: https://medium.com/syntio/streaming-data-from-twitter-to-gcp-7b92c84211a7


## Generate Random Climate Facts

The Climate AIdvocate regularly asks for GPT-3 to generate a fun fact about a
curated list of topics. This is also implemented via a Cloud Function with a
Cloud Trigger that runs on a fixed cadence and a RNG roll to prevent it from
posting at that fixed cadence.

The RNG was just a cheap way to let the bot tweet in a less predictable cadence
for the duration of the Hackathon.

## Responding to Tweets

The bot is more careful when responding to tweets than when generating fun facts.

It starts by using a fine-tuned GPT-3 model to classify incoming tweets into
`believer`, `neutral` and `den` (denialist).

When addressing believers, it uses a database of curated topics from
[Regeneration's Nexus] to seed the prompt and reply in a way that encourages
believers from taking action. It uses the `text-similarity-davinci-001` model to
generate embeddings for the incoming tweet and cosine similarity to find the
topic in our database that is the closest to the topic in the tweet.

For tweets with neutral or denialist stances it gives a generic response which
tries to reply to potentially misleading topics with factual information.

Before it tweets a response, it uses the same classifier that was used on the
incoming tweets and if the stance of the reply is not at least neutral, it flags
the tweet as unsafe and doesn't send it.

Every generated response, regardless of whether it was submitted or not, is
logged to another table in BigQuery. This is an important step that allows
humans to monitor the bot activity. It also generates additional training data
for the bot, which can then be use to improve the quality of its answers.

[Regeneration's Nexus]: https://regeneration.org/nexus
