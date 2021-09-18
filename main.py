import yaml

from data_producers.bitcoin_sentiment_dataproducer import BitcoinSentimentDataProducer
from data_producers.bitcoin_price_dataproducer import BitcoinPriceDataProducer
from data_loaders.bitcoin_sentiment_dataloader import BitcoinSentimentDataLoader
from data_loaders.bitcoin_price_dataloader import BitcoinPriceDataLoader
from optimizers.bitcoin_sentiment_optimizer import BitcoinSentimentOptimizer
from optimizers.bitcoin_price_regression_optimizer import BitcoinPriceRegressionOptimizer
from optimizers.bitcoin_price_classification_optimizer import BitcoinPriceClassificationOptimizer
from models.bitcoin_sentiment_models import BitcoinSentimentLSTM1Model, BitcoinSentimentCNN1Model, BitcoinSentimentCNN2Model
from models.bitcoin_price_regression_models import BitcoinPriceLSTMRegressionModel, BitcoinPriceCNNRegressionModel
from models.bitcoin_price_classification_models import BitcoinPriceLSTMClassificationModel, BitcoinPriceCNNClassificationModel
from trainers.bitcoin_sentiment_trainer import BitcoinSentimentModelTrainer
from trainers.bitcoin_price_trainer import BitcoinPriceModelTrainer
from predictors.bitcoin_sentiment_predictor import BitcoinSentimentModelPredictor

from dotmap import DotMap

import numpy as np
import pandas as pd

import click

def get_config():
    with open(r'config.yaml') as file:
        config = yaml.safe_load(file, Loader=yaml.FullLoader)

    return config

@click.group()
def cli1():
    pass

@cli1.command()
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
def sentiment_data_retrieve(pull_data):
    
    config = DotMap()
    config.dataproducer.pull_data = pull_data
    config.apikey = get_config()["crypto_api_key"]

    print('Create the data producer.')
    data_producer = BitcoinSentimentDataProducer(config)

    df_raw = data_producer.get_raw_data()
    df_processed = data_producer.get_processed_data()

    print(df_raw.head())
    print(df_processed.head())

@click.group()
def cli2():
    pass

@cli2.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='', type=str)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
@click.option('--n-trials', help='number of study trials. Default is 2', default=2, type=int)
@click.option('--version', help='version of study. Default is 1', default=1, type=int)
def sentiment_optimize(model_type, pull_data, n_trials, version):
    
    config = DotMap()
    print(pull_data)
    config.dataloader.pull_data = pull_data
    config.trainer.model_type = model_type
    config.optimizer.n_trials = n_trials
    config.optimizer.version = version
    config.trainer.model_name_prefix = './artifacts/sent'

    print('Create the data generator.')
    data_loader = BitcoinSentimentDataLoader(config)

    optimizer = BitcoinSentimentOptimizer(data_loader.get_train_data(), data_loader.get_test_data(), config)

    optimizer.optimize()

@click.group()
def cli3():
    pass

@cli3.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='LSTM', type=str)
@click.option('--epochs', help='Epochs', default=10, type=int)
@click.option('--batch-size', help='Epochs', default=10, type=int)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
def sentiment_train(model_type, epochs, batch_size, pull_data):
    
    config = DotMap()
    config.pull_data = pull_data
    config.trainer.model_type = model_type
    config.trainer.num_epochs = epochs
    config.trainer.verbose_training = 1
    config.trainer.batch_size = batch_size
    config.trainer.optimizer_name = 'adam'
    config.trainer.validation_split = .1
    config.trainer.max_article_length = 800
    config.trainer.embedding_vector_length = 100
    config.trainer.model_name_prefix = './artifacts/sent'
    config.trainer.model_name = "{}-{}.h5".format(config.trainer.model_name_prefix, model_type)

    print('Create the data generator.')
    data_loader = BitcoinSentimentDataLoader(config)

    print('Create the model.')
    if model_type == 'LSTM':
        model = BitcoinSentimentLSTM1Model(config)
    else:
        model = BitcoinSentimentCNN1Model(config)

    print('Create the trainer')
    trainer = BitcoinSentimentModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_test_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Start training the model.')
    trainer.evaluate()

@click.group()
def cli4():
    pass

@cli4.command()
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
def price_data_retrieve(pull_data):
    
    config = DotMap()
    config.dataproducer.pull_data = pull_data

    print('Create the data producer.')
    data_producer = BitcoinPriceDataProducer(config)

    df_raw = data_producer.get_raw_data()
    df_processed = data_producer.get_processed_data()

    print(df_raw.head())
    print(df_processed.head())


@click.group()
def cli5():
    pass

@cli5.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='', type=str)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
@click.option('--n-trials', help='number of study trials. Default is 2', default=2, type=int)
@click.option('--with-sent', help='With or without sentiment feature. Default is True', default=True, type=bool)
@click.option('--version', help='version of study. Default is 1', default=1, type=int)
def price_regression_optimize(model_type, pull_data, n_trials, with_sent, version):
    
    config = DotMap()
    config.pull_data = pull_data
    config.trainer.model_type = model_type
    config.trainer.model_task = 'regression'
    config.optimizer.n_trials = n_trials
    config.optimizer.version = version
    config.trainer.model_name_prefix = './artifacts/reg'
    config.trainer.with_sent = with_sent

    print('Create the data generator.')
    data_loader = BitcoinPriceDataLoader(config)
    config.trainer.nfeatures = int(data_loader.get_train_data()[0].shape[1])
    print(config.trainer.nfeatures)

    optimizer = BitcoinPriceRegressionOptimizer(data_loader.get_train_data(), data_loader.get_test_data(), config)

    optimizer.optimize()

@click.group()
def cli6():
    pass

@cli6.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='LSTM', type=str)
@click.option('--epochs', help='Epochs', default=10, type=int)
@click.option('--batch-size', help='Epochs', default=10, type=int)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
@click.option('--with-sent', help='With or without sentiment feature. Default is True', default=True, type=bool)
def price_regression_train(model_type, epochs, batch_size, pull_data, with_sent):
    
    config = DotMap()
    config.pull_data = pull_data
    config.trainer.model_type = model_type
    config.trainer.model_task = 'regression'
    config.trainer.num_epochs = epochs
    config.trainer.verbose_training = 1
    config.trainer.batch_size = batch_size
    config.trainer.units = 32
    config.trainer.activation_function = 'relu'
    config.trainer.optimizer_name = 'adam'
    config.trainer.validation_split = .1
    config.trainer.model_name_prefix = './artifacts/reg'
    config.trainer.with_sent = with_sent
    config.trainer.model_name = "{}-{}.h5".format(config.trainer.model_name_prefix, model_type)

    print('Create the data generator.')
    data_loader = BitcoinPriceDataLoader(config)
    config.trainer.nfeatures = int(data_loader.get_train_data()[0].shape[1])

    print('Create the model.')
    if model_type == 'LSTM':
        model = BitcoinPriceLSTMRegressionModel(config)
    else:
        model = BitcoinPriceCNNRegressionModel(config)

    print('Create the trainer')
    trainer = BitcoinPriceModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_test_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Start training the model.')
    trainer.evaluate()

@click.group()
def cli7():
    pass

@cli7.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='', type=str)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
@click.option('--n-trials', help='number of study trials. Default is 2', default=2, type=int)
@click.option('--with-sent', help='With or without sentiment feature. Default is True', default=True, type=bool)
@click.option('--version', help='version of study. Default is 1', default=1, type=int)
def price_classification_optimize(model_type, pull_data, n_trials, with_sent, version):
    
    config = DotMap()
    config.pull_data = pull_data
    config.trainer.model_type = model_type
    config.trainer.model_task = 'classification'
    config.optimizer.n_trials = n_trials
    config.optimizer.version = version
    config.trainer.model_name_prefix = './artifacts/cls'
    config.trainer.with_sent = with_sent

    print('Create the data generator.')
    data_loader = BitcoinPriceDataLoader(config)
    config.trainer.nfeatures = int(data_loader.get_train_data()[0].shape[1])

    optimizer = BitcoinPriceClassificationOptimizer(data_loader.get_train_data(), data_loader.get_test_data(), config)

    optimizer.optimize()

@click.group()
def cli8():
    pass

@cli8.command()
@click.option('--model-type', help='LSTM, CNN. Default uses all', default='LSTM', type=str)
@click.option('--epochs', help='Epochs', default=10, type=int)
@click.option('--batch-size', help='Epochs', default=10, type=int)
@click.option('--pull-data', help='load saved data or pull new data. Default is False', default=False, type=bool)
@click.option('--with-sent', help='With or without sentiment feature. Default is True', default=True, type=bool)
def price_classification_train(model_type, epochs, batch_size, pull_data, with_sent):
    
    config = DotMap()
    config.pull_data = pull_data
    config.trainer.model_type = model_type
    config.trainer.model_task = 'classification'
    config.trainer.num_epochs = epochs
    config.trainer.verbose_training = 1
    config.trainer.batch_size = batch_size
    config.trainer.units = 32
    config.trainer.activation_function = 'relu'
    config.trainer.optimizer_name = 'adam'
    config.trainer.validation_split = .1
    config.trainer.model_name_prefix = './artifacts/cls'
    config.trainer.with_sent = with_sent
    config.trainer.model_name = "{}-{}.h5".format(config.trainer.model_name_prefix, model_type)

    print('Create the data generator.')
    data_loader = BitcoinPriceDataLoader(config)
    config.trainer.nfeatures = int(data_loader.get_train_data()[0].shape[1])

    print('Create the model.')
    if model_type == 'LSTM':
        model = BitcoinPriceLSTMClassificationModel(config)
    else:
        model = BitcoinPriceCNNClassificationModel(config)

    print('Create the trainer')
    trainer = BitcoinPriceModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_test_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Start training the model.')
    trainer.evaluate()

@click.group()
def cli9():
    pass

@cli9.command()
def sentiment_predict():

    config = DotMap()

    predictor = BitcoinSentimentModelPredictor(config)

    prediction = predictor.predict()
    data = predictor.get_data()

    df_news = pd.read_pickle('./data/processed/news_corpus_predict.pkl')
    df_news['polarity'] = prediction
    print(df_news.tail())
    print(df_news.info())
    df_news['polarity'] = pd.to_numeric(df_news['polarity'])
    df_news = df_news.groupby(['date'])['polarity'].mean()
    df_news = df_news.reset_index('date')
    df_news.set_index('date', inplace=True)
    df_news.sort_index(inplace=True)

    df_chartdata = pd.read_pickle('./data/processed/chartdata.pkl')
    df_chartdata_sent = df_chartdata.merge(df_news[['polarity']], on="date", how="inner")
    df_chartdata_sent.to_pickle('./data/processed/chartdata_sent.pkl')

    print(prediction.flatten())


cli = click.CommandCollection(sources=[cli1, cli2, cli3, cli4, cli5, cli6, cli7, cli8, cli9])

if __name__ == '__main__':
    cli()