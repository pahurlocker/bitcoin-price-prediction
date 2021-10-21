# Bitcoin Price Prediction

## Overview

This repository contains code used in a study to predict the price of Bitcoin using various blockchain features and news sentiment. The experiment is focused on determining the accuracy of Bitcoin price prediciton with and without sentiment. The corpus for the sentiment analysis is pulled from various news sources. The project includes various deep learning model architectures to perform sentiment analysis and predict Bitcoin prices using different size time steps. There are two types of Bitcoin prediction models, the regression models attempt to predict the price and the classification models attempt to predict whether the price will increase or decrease.

## Project Structure

All methods can be executed via the command line using `main.py`. Rubicon is used for experiment tracking and Optuna is used for hyperparameter optimization.

```nohighlight
├── artifacts                               <- trained models, tokenizers, scalers, and optuna studies
    ├── rubicon-root                        <- rubicon experiment results
├── base                                    <- base classes 
├── data                                    <- bitcoin chart data and news corpus
    ├── processed                           <- cleansed, transformed, scaled data
    ├── raw                                 <- raw chart and news data from apis 
├── data_loaders                            <- classes that load preprocessed data and splits it 
├── data_producers                          <- classes that load data from apis and preprocesses data
├── models                                  <- price and sentiment models
├── optimizers                              <- optuna optimizers for price and sentiment models
├── predictors                              <- model prediction classes
├── trainers                                <- model training classes
├── bitcoin-price-prediction-eda.ipynb      <- Jupyter notebook for project eda
├── main.py                                 <- command line methods
├── price-prediction-experiments.sh         <- script for price prediction experiments
├── requirements.txt                        <- conda environment requirements
```

## Data Sources

### Crypto New API

The news articles used to produce the corpus were obtained from cryptonews-api.com. An API key must be obtained from them and placed in a config.yaml file in the root of the project.

### Blockchain<span>.co</span>m API

The blockchain metrics are obtained from the blockchain.com API. More information about the information is available at https://www.blockchain.com/charts. An API key is not required.


## Model Pipelines

The command line methods below represent the steps in the model pipeline for retrieving the news corpus and blockchain chart data, training the sentiment model using hyperparameter optimization, performing sentiment predictions on the hold out for Bitcoin price prediction, and training the regression and classification Bitcoin price prediction models.

### Retrieve Data

```
Usage: main.py sentiment-data-retrieve [OPTIONS]

Options:
  --pull-data BOOLEAN  load saved data or pull new data. Default is False
  --help               Show this message and exit.
```

```
Usage: main.py price-data-retrieve [OPTIONS]

Options:
  --pull-data BOOLEAN  load saved data or pull new data. Default is False
  --help               Show this message and exit.
```

### Sentiment Analysis Model Training

```
Usage: main.py sentiment-optimize [OPTIONS]

Options:
  --model-type TEXT    LSTM, CNN. Default uses all
  --pull-data BOOLEAN  load saved data or pull new data. Default is False
  --n-trials INTEGER   number of study trials. Default is 2
  --version INTEGER    version of study. Default is 1
  --help               Show this message and exit.
```

### Sentiment Model Prediction

```
Usage: main.py sentiment-predict [OPTIONS]

Options:
  --help  Show this message and exit.
```

### Price Regression Model Training with Hyperparameter Optimization

```
Usage: main.py price-regression-optimize [OPTIONS]

Options:
  --model-type TEXT            LSTM, CNN. Default uses all
  --pull-data BOOLEAN          load saved data or pull new data. Default is
                               False
  --n-trials INTEGER           number of study trials. Default is 2
  --with-sent BOOLEAN          With or without sentiment feature. Default is
                               True
  --timestep INTEGER           Number of timesteps, if not set the optimizer
                               will select the timesteps. Default is 0
  --prediction-length INTEGER  The number of predictions in the test set. The
                               size of the test set will be prediction length
                               + timesteps + 1. Be careful not to go above the
                               total size of data set. Default is 5
  --version INTEGER            version of study. Default is 1
  --help                       Show this message and exit.
```

### Price Regression Model Training of Individual Model

```
Usage: main.py price-regression-train [OPTIONS]

Options:
  --model-type TEXT     LSTM, CNN. Default uses all
  --epochs INTEGER      Epochs
  --batch-size INTEGER  Batch size
  --pull-data BOOLEAN   load saved data or pull new data. Default is False
  --with-sent BOOLEAN   With or without sentiment feature. Default is True
  --help                Show this message and exit.
```

### Price Classification Model Training with Hyperparameter Optimization

```
Usage: main.py price-classification-optimize [OPTIONS]

Options:
  --model-type TEXT            LSTM, CNN. Default uses all
  --pull-data BOOLEAN          load saved data or pull new data. Default is
                               False
  --n-trials INTEGER           number of study trials. Default is 2
  --with-sent BOOLEAN          With or without sentiment feature. Default is
                               True
  --timestep INTEGER           Number of timesteps, if not set the optimizer
                               will select the timesteps. Default is 0
  --prediction-length INTEGER  The number of predictions in the test set. The
                               size of the test set will be prediction length
                               + timesteps + 1. Be careful not to go above the
                               total size of data set. Default is 5
  --version INTEGER            version of study. Default is 1
  --help                       Show this message and exit.
```

### Price Classification Model Training of Individual Model

```
Usage: main.py price-classification-train [OPTIONS]

Options:
  --model-type TEXT     LSTM, CNN. Default uses all
  --epochs INTEGER      Epochs
  --batch-size INTEGER  Batch size
  --pull-data BOOLEAN   load saved data or pull new data. Default is False
  --with-sent BOOLEAN   With or without sentiment feature. Default is True
  --help                Show this message and exit.
```

## Price Prediction Experiments

The following scripts can be run to replicate the price regression and classification experiments:

```bash
./price-prediction-regression-experiment.sh
./price-prediction-classification-experiment.sh
```

## Tools

### Running Optuna Dashboard

```bash
optuna-dashboard sqlite:///artifacts/db.sqlite3    
```

### Running Rubicon

```bash
rubicon_ml ui --root-dir ./artifacts/rubicon-root
```