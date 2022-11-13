import numpy as np
import openai
import tweepy
from google.cloud import secretmanager
from tenacity import retry, stop_after_attempt, wait_random_exponential

TOPICS = [
    "afforestation",
    "bamboo",
    "boreal forests",
    "carbon capture",
    "composting",
    "distributed energy",
    "eating plants",
    "education",
    "electric vehicles",
    "electrification",
    "marine protected areas",
    "micromobility",
    "ocean farming",
    "protecting coral reefs",
    "protecting forests",
    "protecting insects",
    "protecting mangroves",
    "protecting oceans",
    "protecting wetlands",
    "regenerative agriculture",
    "renewable energy",
    "seaforestation",
    "seagrasses and mangroves",
    "solar energy",
    "sustainable agriculture",
    "tropical forests",
    "urban agriculture",
    "wasting less food",
    "wind energy",
    "woman led agriculture",
]


def authenticate_openai() -> None:
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


def tweet_fun_fact(text: str):
    client = new_tweepy_client()
    return client.create_tweet(text=text)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def complete_fun_fact(topic: str, max_tokens: int = 280, temperature: int = 0.5) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"In less than 280 characters, write a fun fact about how {topic} can help reduce the impact of climate change:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def generate_fun_fact(request):
    # Only run this function half of the time.
    if np.random.random() < 0.5:
        return ("", 204)

    authenticate_openai()

    topic = np.random.choice(TOPICS)
    text = complete_fun_fact(topic)
    hashtag = "\n#" + topic.replace(" ", "")
    if len(text) + len(hashtag) < 280:
        text += hashtag

    if text is not None:
        response = tweet_fun_fact(text)

    resp = ""
    if response and response.data["id"] is not None:
        resp = response.data["id"]

    return (resp, 204)
