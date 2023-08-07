# River Batch Example

This is an example of a sentiment analysis application using sample yelp ratings data from [Kaggle](https://www.kaggle.com) using [River](https://riverml.xyz/0.18.0/) and [PyEnsign](https://github.com/rotationalio/pyensign).  This is a modified version of the code under the `River` directory in this repo.  This example walks through a scenario where an online model is incrementally trained on a subset of data and then published to a topic, which is then used by a downstream application that will use this model to generate predictions.

Publishing the model and its corresponding to a topic allows for a seamless way to do ML Ops.  Models can be version controlled and retrieved from the topic as needed.  The performance of the different models can also be compared by looking at the metrics corresponding to each model version.

In order to use PyEnsign, create a free account on [Rotational.app](https://rotational.app/), generate and download API Keys.  You will need to create and source the following environment variables prior to running the example:

```
export ENSIGN_CLIENT_ID="your client id here"
export ENSIGN_CLIENT_SECRET="your client secret here"
```

This application consists of four components:
- `TradeDataPublisher` reads data from the `yelp_train.csv` file and publishes to the `river_train_data` topic.  Note that this csv can easily be replaced by a real-time data source.  Check out Rotational Labs [Data Playground](https://github.com/rotationalio/data-playground) for examples!
- `Trainer` listens for new messages on the `river_train_data` topic and builds an online model that learns incrementally as it receives new training instances.  Once it is done training, it publishes the final model and metrics to the `river_model` topic.
- `ScoreDataPublisher` reads data from the `yelp_score.csv` file publishes to the `river_score_data` topic.
- `Scorer` listens for new messages in the `river_model` and `river_score_data` topics.  When it receives a message from the `river_model` topic, it loads the model which it then uses to make predictions on data that it receives from the `river_score_data` topic.

## Steps to run the application

### Create a virtual environment

```
$ virtualenv venv
```

### Activate the virtual environment

```
$ source venv/bin/activate
```

### Install the required packages

```
$ pip install -r requirements.txt
```

### Open four terminal windows

#### Run the Scorer in the first window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```

```
$ python river_sentiment_analysis_batch.py score
```

#### Run the Trainer in the second window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_analysis_batch.py train
```

#### Run the TrainDataPublisher in the third window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_analysis_batch.py train_data
```

#### Run the Scorer in the fourth window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_analysis_batch.py score_data
```