import os
import sys
import asyncio
import json
from datetime import datetime

import pandas as pd
from pyensign.events import Event
from pyensign.ensign import Ensign
from river.cluster import STREAMKMeans


async def handle_ack(ack):
    ts = datetime.fromtimestamp(ack.committed.seconds + ack.committed.nanos / 1e9)
    print(ts)

async def handle_nack(nack):
    print(f"Could not commit event {nack.id} with error {nack.code}: {nack.error}")

 
class HousingDataPublisher:
    def __init__(self, topic="housing-json"):
        self.topic = topic
        self.ensign = Ensign()

    def run(self):
        """
        Run the publisher
        """
        asyncio.run(self.publish())

    async def publish(self):
        """
        Read data from the housging.csv file and publish to housing-json topic.
        This can be replaced by a real time streaming source
        Check out https://github.com/rotationalio/data-playground for examples
        """
        # create the topic if it does not exist
        await self.ensign.ensure_topic_exists(self.topic)
        train_df = pd.read_csv(os.path.join("data", "housing.csv"))
        train_dict = train_df.to_dict("records")
        for record in train_dict:
            print(record)
            event = Event(json.dumps(record).encode("utf-8"), mimetype="application/json")
            await self.ensign.publish(self.topic, event, on_ack=handle_ack, on_nack=handle_nack)


class HousingDataSubscriber:
    """
    The HousingDataSubscriber class reads from the housing-json topic and incrementally 
    learns from the data and generates clusters.
    """

    def __init__(self, topic="housing-json"):
        self.topic = topic
        self.ensign = Ensign()
        self.initialize_model()

    def run(self):
        """
        Run the subscriber
        """
        asyncio.run(self.subscribe())

    def initialize_model(self):
        """
        Initialize a river clustering model
        Update the parameters as needed for your project
        """
        self.model = STREAMKMeans(chunk_size=3, n_clusters=2, halflife=0.5, sigma=1.5, seed=0)

    async def run_model_pipeline(self, event):
        """
        Incrementally predict and learn/update model
        """
        record = json.loads(event.data)
        print(record)
        # if this is the first time the model is seeing the data, there will be no value for cluster centers
        if len(self.model.centers) > 1:
            cluster = self.model.predict_one(record)
            print(cluster)
        self.model = self.model.learn_one(record)

    async def subscribe(self):
        """
        Receive messages from housing-json topic
        """
        # ensure that the topic exists or create it if it doesn't
        await self.ensign.ensure_topic_exists(self.topic)

        async for event in self.ensign.subscribe(self.topic):
            await self.run_model_pipeline(event)
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "publish":
            publisher = HousingDataPublisher()
            publisher.run()
        elif sys.argv[1] == "subscribe":
            subscriber = HousingDataSubscriber()
            subscriber.run()
        else:
            print("Usage: python river_clustering.py [publish|subscribe]")
    else:
        print("Usage: python river_clustering.py [publish|subscribe]")