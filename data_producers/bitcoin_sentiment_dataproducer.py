from base.base_data_producer import BaseDataProducer

from newspaper import Article
from newspaper import Config

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

import re

class BitcoinSentimentDataProducer(BaseDataProducer):
    def __init__(self, config):
        super(BitcoinSentimentDataProducer, self).__init__(config)

    def _retrieve_data(self):

        if self.config.dataproducer.pull_data:
            print("Pull data...")
            append_data = []
            for i in range(1, 200):
                df = pd.read_json('https://cryptonews-api.com/api/v1?tickers=BTC&date=yeartodate&items=50&token='+self.config.apikey+'&page='+str(
                    i))
                append_data.append(df)

            for i in range(1, 200):
                df = pd.read_json('https://cryptonews-api.com/api/v1?tickers=BTC&date=05052021-today&items=50&token='+self.config.apikey+'&page='+str(
                    i))
                append_data.append(df)
            
            df = pd.concat(append_data)
            df.to_pickle('./data/raw/corpus.pkl')
            df = pd.json_normalize(df['data'])
            df.rename(columns={'date':'datetime'}, inplace=True)
            df['date'] = pd.to_datetime(df['datetime'], format='%a, %d %b %Y %H:%M:%S %z', utc=True)
            df['date'] = df['date'].dt.date.astype('datetime64') 
            df.drop(['image_url','topics','tickers'], axis=1, inplace=True)
            df = df[df.type.isin(['Article'])]
            
            nltk.download('punkt')

            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
            config = Config()
            config.browser_user_agent = user_agent

            list = []
            for ind in df.index:
                dict = {}
                #print(df_all['news_url'][ind])
                article = Article(df['news_url'][ind], config=config)
                article.download()
                try:
                    article.parse()
                    article.nlp()
                    dict['datetime'] = df['datetime'][ind]
                    dict['date'] = df['date'][ind]
                    dict['news_url'] = df['news_url'][ind]
                    dict['title'] = df['title'][ind]
                    dict['text'] = df['text'][ind]
                    dict['source_name'] = df['source_name'][ind]
                    dict['sentiment'] = df['sentiment'][ind]
                    dict['type'] = df['type'][ind]
                    dict['article_title'] = article.title
                    dict['article_text'] = article.text
                    dict['article_summary'] = article.summary
                    list.append(dict)
                except:
                    pass
            
            full_df = pd.DataFrame(list)
            full_df.to_pickle("./data/raw/news_corpus_082021.pkl")
        else:
            print('Read saved data...')
            full_df = pd.read_pickle('./data/raw/news_corpus_082021.pkl')

        return full_df

    def _preprocess_data(self):

        stop_words = stopwords.words('english')
        # Retrieve raw data
        df = pd.read_pickle('./data/raw/news_corpus.pkl')

        sentiment_dict = {'Negative': 0, 'Neutral': 2, 'Positive': 1}
        df['polarity'] = df.sentiment.map(sentiment_dict)
        df.drop(df[df['polarity'] == 2].index, inplace=True)
        reputable_news = ['Forbes', 'CNBC Television', 'Bitcoin',
                      'Yahoo Finance', 'Bloomberg Markets and Finance', 'Reuters', 'Fox Business', 'Bloomberg Technology',
                      'Coindesk', 'CNBC']
        #df = df[df.source_name.isin(reputable_news)]
        df['text'] = df['article_summary']
        df['text'] = df['text'].apply(lambda x: x.lower())
        df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z\s]', ' ', x)))
        df['text'] = df['text'].apply((lambda x: re.sub(r'\d+', '', x)))
        #df['text'] = df['text'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        training_start = int(len(df) * .4)

        df_train = df[['text', 'polarity']].iloc[training_start:]
        df_predict = df.iloc[:training_start]
        # df_train = df[['text', 'polarity']].iloc[:training_start]
        # df_predict = df.iloc[training_start:]

        df_train.to_pickle("./data/processed/news_corpus.pkl")
        df_predict.to_pickle("./data/processed/news_corpus_predict.pkl")

        return df_train

    # def get_raw_data(self):
    #     return pd.read_pickle('./data/raw/news_corpus.pkl')

    # def get_processed_data(self):
    #     return pd.read_pickle('./data/processed/news_corpus.pkl')

