FROM python:3

RUN pip install --upgrade pip
RUN pip install tweepy
RUN pip install --upgrade google-cloud-pubsub
RUN pip install --upgrade google-cloud-secret-manager

ADD twitter_to_pubsub.py /twitter_to_pubsub.py

CMD python twitter_to_pubsub.py
