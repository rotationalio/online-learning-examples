# River

This is an example of a sentiment analysis application using sample yelp ratings data from [Kaggle](https://www.kaggle.com) using [River](https://riverml.xyz/) and [PyEnsign](https://github.com/rotationalio/pyensign).

In order to use PyEnsign, create a free account on [Rotational.app](https://rotational.app/), generate and download API Keys.  You will need to create and source the following environment variables prior to running the example:

```
export ENSIGN_CLIENT_ID="your client id here"
export ENSIGN_CLIENT_SECRET="your client secret here"
```

This application consists of three components:
- `YelpDataPublisher` reads data from the `yelp.csv` file and publishes to the `river_pipeline` topic.  Note that this csv can easily be replaced by a real-time data source.  Check out Rotational Labs [Data Playground](https://github.com/rotationalio/data-playground) for examples!
- `YelpDataSubscriber` listens for new messages on the `river_pipeline` topic and loads and uses an online model that learns incrementally as it receives new training instances.  It also calculates the precision and recall scores which it publishes to the `river_metrics` topic.
- `MetricsSubscriber` listens for new messages on the `river_metrics` topic and checks to see if the precision or recall score are below a pre-specified threshold and prints the values if they fall below the threshold.

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

### Open three terminal windows

#### Run the MetricsSubscriber in the first window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```

```
$ python river_sentiment_analysis.py metrics
```

#### Run the YelpDataSubscriber in the second window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_analysis.py subscribe
```

#### Run the YelpDataPublisher in the third window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_analysis.py publish
```
