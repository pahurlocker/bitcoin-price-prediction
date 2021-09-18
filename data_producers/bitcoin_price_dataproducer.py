from base.base_data_producer import BaseDataProducer

import pandas as pd

import datetime

metrics = ['blocks-size', 'avg-block-size', 'n-transactions-total',
           'hash-rate', 'difficulty', 'transaction-fees-usd',
           'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
           'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
           'trade-volume', 'total-bitcoins', 'market-price']

years = [2020]

class BitcoinPriceDataProducer(BaseDataProducer):
    def __init__(self, config):
        super(BitcoinPriceDataProducer, self).__init__(config)

    def _retrieve_data(self):
        df_all = pd.DataFrame()

        if self.config.dataproducer.pull_data:
            for m in metrics:
                append_data = []
                for y in years:
                    ts = datetime.datetime(
                        y, 12, 31, tzinfo=datetime.timezone.utc).timestamp()
                    print('https://api.blockchain.info/charts/'+m +
                        '?timespan=1year&rollingAverage=24hours&format=csv&start='+str(int(ts)))
                    df = pd.read_csv('https://api.blockchain.info/charts/'+m+'?timespan=1year&rollingAverage=24hours&format=csv&start='+str(
                        int(ts)), names=['date', m], parse_dates=[0], index_col=[0])
                    append_data.append(df)
                df_m = pd.concat(append_data)
                df_m.index = df_m.index.normalize()
                df_m = df_m.groupby([pd.Grouper(freq='D')]).mean()

                if df_all.shape[0] == 0:
                    print(m)
                    print(df_m.shape)
                    df_all = df_m
                else:
                    print(m)
                    print(df_m.shape)
                    print(df_all.shape)
                    df_all = df_all.merge(df_m, on="date", how="outer")

            df_all.to_pickle('./data/raw/chartdata_082021.pkl')
        else:
            df_all = pd.read_pickle('./data/raw/chartdata_082021.pkl')

        return df_all

    def _preprocess_data(self):

        df = pd.read_pickle('./data/raw/chartdata.pkl')
        df.dropna(subset=['market-price'], inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        df.to_pickle("./data/processed/chartdata.pkl")

        return df