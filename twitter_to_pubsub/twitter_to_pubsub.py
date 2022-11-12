#!/usr/bin/env python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script uses the Twitter Streaming API, via the tweepy library,
to pull in tweets and publish them to a PubSub topic.
"""

import datetime
import os
from concurrent import futures

from google.cloud import pubsub_v1, secretmanager
from tweepy import StreamingClient, StreamRule


class PubSubStreamer(StreamingClient):
    """A listener that handles tweets received from a stream.

    This listener batches tweets and publishes them to a PubSub topic.
    """

    count = 0
    total_tweets = 100_000

    batch_settings = pubsub_v1.types.BatchSettings(
        max_messages=50, max_bytes=1024 * 1024, max_latency=10
    )

    client = pubsub_v1.PublisherClient(batch_settings)
    topic_path = client.topic_path(os.environ["PROJECT_ID"], os.environ["PUBSUB_TOPIC"])

    publish_futures = []

    def on_connect(self):
        print("Connected")

    # Resolve the publish future in a separate thread.
    def callback(self, future: pubsub_v1.publisher.futures.Future) -> None:
        _ = future.result()

    def publish_message(self, data):
        publish_future = self.client.publish(self.topic_path, data)
        publish_future.add_done_callback(self.callback)
        self.publish_futures.append(publish_future)
        self.count += 1

    def on_data(self, data):
        self.publish_message(data)

        if self.count > self.total_tweets:
            futures.wait(self.publish_futures, return_when=futures.ALL_COMPLETED)
            return False

        if (self.count % 1000) == 0:
            print("count is: %s at %s" % (self.count, datetime.datetime.now()))
        return True

    def on_error(self, status):
        print(status)


if __name__ == "__main__":
    print("Initializing Twitter Stream")

    secret_client = secretmanager.SecretManagerServiceClient()
    project = f"projects/{os.environ['PROJECT_ID']}"
    secret_name = f"{project}/secrets/TWITTER_BEARER_TOKEN/versions/latest"
    response = secret_client.access_secret_version(request={"name": secret_name})
    bearer_token = response.payload.data.decode("UTF-8")

    stream = PubSubStreamer(bearer_token)

    rules = [
        "#climatechange",
        "#climatechangeisreal",
        "#actonclimate",
        "#globalwarming",
        "#climagechangehoax",
        "#climatedeniers",
        "#climeatechangeisfalse",
        "#globalwarminghoax",
        "#climatechangenotreal",
    ]
    stream.add_rules([StreamRule(rule) for rule in rules])
    stream.filter()
