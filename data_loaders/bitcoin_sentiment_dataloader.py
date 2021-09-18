from base.base_data_loader import BaseDataLoader

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import NearMiss, RandomUnderSampler

import pandas as pd

import pickle

class BitcoinSentimentDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BitcoinSentimentDataLoader, self).__init__(config)
        
        max_article_length = 100

        df = pd.read_pickle('./data/processed/news_corpus.pkl')
        tokenizer = Tokenizer(num_words=10000, split=' ')
        tokenizer.fit_on_texts(df['text'].values)
        with open('./artifacts/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        X = tokenizer.texts_to_sequences(df['text'].values)
        X = sequence.pad_sequences(X, maxlen=max_article_length, padding="post")
        Y = df['polarity'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1, shuffle=True)
        #up-sample data
        # self.X_train, self.y_train = NearMiss(version=1).fit_resample(self.X_train, self.y_train)
        # Pad the sequence to the same length
        #self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_article_length)
        #self.X_test = sequence.pad_sequences(self.X_test, maxlen=max_article_length)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test