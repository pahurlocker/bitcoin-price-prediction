from base.base_predictor import BasePredictor

import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import pickle

class BitcoinSentimentModelPredictor(BasePredictor):
    def __init__(self, config):
        super(BitcoinSentimentModelPredictor, self).__init__(config)

    def _load_model(self):
        return load_model("./artifacts/best-CNN3/sent-final.h5", compile=False)

    def _load_data(self):

        max_article_length = 100

        df = pd.read_pickle('./data/processed/news_corpus_predict.pkl')
        with open('./artifacts/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        # print(tokenizer.word_index)  # To see the dictionary
        X = tokenizer.texts_to_sequences(df['text'].values)
        X = sequence.pad_sequences(X, maxlen=max_article_length, padding="post")
        print(df['polarity'].values.flatten())
        return X

    def predict(self):
        prediction = self.model.predict_classes(self.data)
        
        return prediction

    def get_data(self):
        return self.data