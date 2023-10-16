# River

## Sentiment Analysis
`river_sentiment_analysis.py` is an example of a sentiment analysis application using sample yelp ratings data from [Kaggle](https://www.kaggle.com) using [River](https://riverml.xyz/) and [PyEnsign](https://github.com/rotationalio/pyensign).

To use PyEnsign, create a free account on [rotational.app](https://rotational.app/).  You will need to do the following once you create an account:

- [Create a project.](https://youtu.be/VskNgAVMORQ)
- Add the following topics to the project: `river_pipeline`, `river_metrics`.  Check out this [video](https://youtu.be/1XuVPl_Ki4U) on how to add a topic.  You can choose your own names for the topic but make sure that you update the code accordingly.
- [Generate API keys for your project.](https://youtu.be/KMejrUIouMw)

You will need to create and source the following environment variables using the API keys you just downloaded:


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

## Clustering
`river_clustering.py` is an example of a clustering application that uses sample housing data from [Kaggle](https://www.kaggle.com).  Follow the same steps as above to get started but instead of the topics listed above, you will need to create a new topic called `housing-json`.

This application consists of two components:
- `HousingDataPublisher` reads data from the `housing.csv` file and publishes to the `housing-json` topic.  Note that this csv can easily be replaced by a real-time data source.  Check out Rotational Labs [Data Playground](https://github.com/rotationalio/data-playground) for examples!
- `HousingDataSubscriber` listens for new messages on the `housing-json` topic and loads and uses an online model that learns incrementally as it receives new instances and generates clusters.  NOTE: this example uses the default parameters provided in the River documentation.  In order to generate more accurate clusters, the model parameters will need to be adjusted.

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

### Open two terminal windows

#### Run the HousingDataSubscriber in the first window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_sentiment_clustering.py subscribe
```

#### Run the HousingDataPublisher in the second window (make sure to activate the virtual environment first)
```
$ source venv/bin/activate
```
```
$ python river_clustering.py publish
```
