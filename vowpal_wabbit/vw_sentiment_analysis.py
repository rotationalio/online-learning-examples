import os
import re
import sys
import json
import asyncio
from datetime import datetime

import pandas as pd
from pyensign.events import Event
from pyensign.ensign import Ensign
from sklearn.metrics import confusion_matrix,  precision_score, recall_score
import vowpalwabbit

async def handle_ack(ack):
    ts = datetime.fromtimestamp(ack.committed.seconds + ack.committed.nanos / 1e9)
    print(ts)

async def handle_nack(nack):
    print(f"Could not commit event {nack.id} with error {nack.code}: {nack.error}")

def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

class YelpDataPublisher:
    def __init__(self, topic="vw_pipeline", interval=1):
        self.topic = topic
        self.interval = interval
        self.ensign = Ensign()

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.publish())

    async def publish(self):
        """
        Read data from the yelp_train.csv file and publish to vw_pipeline topic.
        This can be replaced by a real time streaming source
        Check out https://github.com/rotationalio/data-playground for examples
        """
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        train_df = pd.read_csv(os.path.join("data", "yelp.csv"))
        # for binary classification, VW expects the labels to be 1 and -1
        train_df["sentiment"] = train_df["sentiment"].apply(lambda x: 1 if x==1 else -1)
        train_dict = train_df.to_dict("records")
        for record in train_dict:
            print(record)
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)

        event = Event(json.dumps({"done":"yes"}).encode("utf-8"), mimetype="application/json")
        await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)
        await asyncio.sleep(self.interval)
    

class YelpDataSubscriber:
    """
    The YelpDataSubscriber class reads from the vw_pipeline topic and incrementally learns
    from the data until it has learned from all of the instances.  It publishes the precision
    and recall metrics to the vw_metrics topic after they are calculated at each step.
    """

    def __init__(self, sub_topic="vw_pipeline", pub_topic="vw_metrics", interval=1):
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.interval = interval
        self.ensign = Ensign()
        self.initialize_model()

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    def initialize_model(self):
        self.model = vowpalwabbit.Workspace("--loss_function=logistic -b 28 --ngram 3 --binary --quiet")
        self.labels = []
        self.preds = []

    async def run_model_pipeline(self, event):
        """
        Receive messages from the websocket and publish events to Ensign.
        """
        record = json.loads(event.data)
        if "done" not in record.keys():
            text = record["text"]
            # convert the text to vw format (only pass in the text here and see how vw predicts)
            train_instance = to_vw_format(text)
            y_pred = int(self.model.predict(train_instance))
            self.preds.append(y_pred)
            label = record["sentiment"]
            self.labels.append(label)
            # the precision and recall won't be great at first, but as the model learns on
            # new data, the scores improve
            precision = precision_score(self.labels, self.preds, pos_label=-1, average="binary")
            print(f"Precision: {precision}")
            recall = recall_score(self.labels, self.preds, pos_label=-1, average="binary")
            print(f"Recall: {recall}")
            pr_dict = {"precision": precision, "recall": recall}
            event = Event(json.dumps(pr_dict).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.pub_topic, event, on_ack=handle_ack, on_nack=handle_nack)

            # pass the text and label this time so that the model can learn from the example
            learn_instance = to_vw_format(text, label)
            self.model.learn(learn_instance)
        else:
            # We are printing out the final metrics here because we have looped through all of 
            # the records.
            print("Final Metrics")
            precision = precision_score(self.labels, self.preds, pos_label=-1, average="binary")
            print(f"Precision: {precision}")
            recall = recall_score(self.labels, self.preds, pos_label=-1, average="binary")
            print(f"Recall: {recall}")
            cm = confusion_matrix(self.labels, self.preds)
            print(cm)

    async def subscribe(self):
        """
        Receive messages from river_pipeline topic
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.sub_topic)
        await self.ensign.ensure_topic_exists(self.pub_topic)

        async for event in self.ensign.subscribe(self.sub_topic):
            await self.run_model_pipeline(event)


class MetricsSubscriber:
    """
    The MetricsSubscriber class reads from the vw_metrics topic and checks to see
    if the precision and recall have fallen below a specified threshold and prints to screen.
    This code can be extended to update a dashboard and/or send alerts.
    """

    def __init__(self, topic="vw_metrics", threshold=0.60, interval=1):
        self.topic = topic
        self.interval = interval
        self.threshold = threshold
        self.ensign = Ensign()

    def run(self):
        """
        Run the subscriber forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    async def check_metrics(self, event):
        """
        Check precision and recall metrics and print if below threshold
        """
        metric_info = json.loads(event.data)
        precision = metric_info["precision"]
        recall = metric_info["recall"]
        if precision < self.threshold:
            print(f"Precision is below threshold: {precision}")
        if recall < self.threshold:
            print(f"Recall is below threshold: {recall}")

    async def subscribe(self):
        """
        Receive messages from river_train_data topic
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.topic)

        async for event in self.ensign.subscribe(self.topic):
            await self.check_metrics(event)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "publish":
            publisher = YelpDataPublisher()
            publisher.run()
        elif sys.argv[1] == "subscribe":
            subscriber = YelpDataSubscriber()
            subscriber.run()
        elif sys.argv[1] == "metrics":
            subscriber = MetricsSubscriber()
            subscriber.run()
        else:
            print("Usage: python vw_sentiment_analysis.py [publish|subscribe|metrics]")
    else:
        print("Usage: python vw_sentiment_analysis.py [publish|subscribe|metrics]")