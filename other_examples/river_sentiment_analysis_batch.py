import os
import sys
import asyncio
import json
import pickle
import gzip
from datetime import datetime

import pandas as pd
from pyensign.events import Event
from pyensign.ensign import Ensign
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords
from river.compose import Pipeline
from river import metrics


async def handle_ack(ack):
    ts = datetime.fromtimestamp(ack.committed.seconds + ack.committed.nanos / 1e9)
    print(ts)

async def handle_nack(nack):
    print(f"Could not commit event {nack.id} with error {nack.code}: {nack.error}")

 
class TrainDataPublisher:
    def __init__(self, topic="river_train_data", interval=1):
        self.topic = topic
        self.ensign = Ensign()
        self.interval = interval

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.publish())

    async def publish(self):
        """
        Read data from the yelp_train.csv file and publish to river_train_data topic.
        This can be replaced by a real time streaming source
        Check out https://github.com/rotationalio/data-playground for examples
        """
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        train_df = pd.read_csv(os.path.join("data", "yelp_train.csv"))
        train_dict = train_df.to_dict("records")
        for record in train_dict:
            print(record)
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)

        # In this example, we have reached the end of the file, so we will send a "done" message
        # and the subscriber will use this message as a notification to print out the final statistics.
        # In a production environment, this stream would be open forever, sending messages to the
        # subscriber and the model and metrics will continually get updated.
        event = Event(json.dumps({"done":"yes"}).encode("utf-8"), mimetype="application/json")
        await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)
        await asyncio.sleep(self.interval)

        
class ScoreDataPublisher:
    def __init__(self, topic="river_score_data", interval=1):
        self.topic = topic
        self.ensign = Ensign()
        self.interval = interval

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.publish())

    async def publish(self):
        """
        Read data from the yelp_score.csv file and publish to river_score_data topic.
        This can be replaced by a real time streaming source
        Check out https://github.com/rotationalio/data-playground for examples
        """
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        score_df = pd.read_csv(os.path.join("data", "yelp_score.csv"))
        score_dict = score_df.to_dict("records")
        for record in score_dict:
            print(record)
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)
        
        event = Event(json.dumps({"done":"yes"}).encode("utf-8"), mimetype="application/json")
        await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)
        await asyncio.sleep(self.interval)


class Trainer:
    """
    The Trainer class reads from the river_train_data topic and incrementally learns
    from the data until it has learned from all of the training instances.
    It then publishes the model and results to the river_model topic.
    """

    def __init__(self, sub_topic="river_train_data", pub_topic="river_model", interval=1):
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.ensign = Ensign()
        self.initialize_model_and_metrics()
        self.interval = interval

    def run(self):
        """
        Run the publisher forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    def initialize_model_and_metrics(self):
        """
        Initialize a river model and set up metrics to evaluate the model as it learns
        """
        self.model = Pipeline(('vectorizer', BagOfWords(lowercase=True)),('nb', MultinomialNB()))
        self.confusion_matrix = metrics.ConfusionMatrix(classes=[0,1])
        self.classification_report = metrics.ClassificationReport()
        self.precision_recall =  metrics.Precision(cm=self.confusion_matrix, pos_val=0) + metrics.Recall(cm=self.confusion_matrix, pos_val=0)

    async def run_model_pipeline(self, event):
        """
        Make a prediction and update metrics based on the predicted value and the actual value
        Incrementally learn/update model based on the actual value
        Continue until "done" message is received
        Publish model and results to river_model topic
        """
        record = json.loads(event.data)
        print(record)
        if "done" not in record.keys():
            y_pred = self.model.predict_one(record["text"])
            if y_pred is not None:
                self.confusion_matrix.update(y_true=record["sentiment"], y_pred=y_pred)
                self.classification_report.update(y_true=record["sentiment"], y_pred=y_pred)
            print(self.precision_recall)
            # learn from the train example and update the model
            self.model = self.model.learn_one(record["text"], record["sentiment"]) 
        else:
            print("Final Metrics", self.precision_recall)
            print(self.classification_report)
            print(self.confusion_matrix)

            model_info = dict()
            model_info["model"] = self.model
            model_info["classification_report"] = self.classification_report
            model_info["confusion_matrix"] = self.confusion_matrix
            compressed_model = gzip.compress(pickle.dumps(model_info))
            meta = {"id": datetime.now().strftime("%Y%m%d%H%M%S")}
            event = Event(compressed_model, mimetype="application/python-pickle", meta=meta)
            print(event.meta)
            await self.ensign.publish(self.pub_topic, event, on_ack=handle_ack, on_nack=handle_nack)
            await asyncio.sleep(self.interval)

    async def subscribe(self):
        """
        Receive messages from river_train_data topic
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.sub_topic)
        await self.ensign.ensure_topic_exists(self.pub_topic)

        async for event in self.ensign.subscribe(self.sub_topic):
            await self.run_model_pipeline(event)


class Scorer:
    """
    Scorer listens to two topics: river_model and river_score_data.
    When it receives a message in the river_model topic, it extracts
    the model from the message.
    Once new data arrives in the river_score_data topic, it uses
    the model to make predictions.  Since this data also contains labels,
    it also updates the metrics.
    """

    def __init__(self, model_topic="river_model", data_topic="river_score_data"):
        self.model_topic = model_topic
        self.data_topic = data_topic
        self.ensign = Ensign()
        self.initialize_metrics()
    
    def run(self):
        """
        Run the subscriber forever.
        """
        asyncio.get_event_loop().run_until_complete(self.subscribe())

    def initialize_metrics(self):
        self.confusion_matrix = metrics.ConfusionMatrix(classes=[0,1])
        self.classification_report = metrics.ClassificationReport()
        self.precision_recall =  metrics.Precision(cm=self.confusion_matrix, pos_val=0) + metrics.Recall(cm=self.confusion_matrix, pos_val=0)

    async def generate_predictions(self, event):
        """
        make a prediction and update metrics until receiving "done" message
        """
        record = json.loads(event.data)
        print(record)
        if "done" not in record.keys():
            print(record)
            y_pred = self.model.predict_one(record["text"])
            print(y_pred)
            if y_pred is not None:
                self.confusion_matrix.update(y_true=record["sentiment"], y_pred=y_pred)
                self.classification_report.update(y_true=record["sentiment"], y_pred=y_pred)
        else:
            print("Final Metrics", self.precision_recall)
            print(self.classification_report)
            print(self.confusion_matrix)


    async def load_model(self, event):
        """
        Train an online model and publish predictions to a new topic.
        Run your super smart model pipeline here!
        """
        print(event.meta)
        data = gzip.decompress(event.data)
        self.model = pickle.loads(data)["model"]

        # listen for new messages on the score_data topic to generate predictions
        async for event in self.ensign.subscribe(self.data_topic):
            await self.generate_predictions(event)

            
    async def subscribe(self):
        """
        Subscribe to events on the river_model and river_score_data topics
        """

        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.model_topic)
        await self.ensign.ensure_topic_exists(self.data_topic)

        # listen for new messages on the model topic
        async for event in self.ensign.subscribe(self.model_topic):
            await self.load_model(event)


if __name__ == "__main__":
    # Run the publisher or subscriber depending on the command line arguments.
    if len(sys.argv) > 1:
        if sys.argv[1] == "train_data":
            publisher = TrainDataPublisher()
            publisher.run()
        elif sys.argv[1] == "train":
            subscriber = Trainer()
            subscriber.run()
        elif sys.argv[1] == "score_data":
            publisher = ScoreDataPublisher()
            publisher.run()
        elif sys.argv[1] == "score":
            subscriber = Scorer()
            subscriber.run()
        else:
            print("Usage: python river_sentiment_analysis_batch.py [train_data|train|score_data|score]")
    else:
        print("Usage: python river_sentiment_analysis_batch.py [train_data|train|score_data|score]")