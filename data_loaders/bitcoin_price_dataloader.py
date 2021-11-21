from base.base_data_loader import BaseDataLoader
from sklearn.model_selection import train_test_split

import pandas as pd

metrics = [
    "blocks-size",
    "avg-block-size",
    "n-transactions-total",
    "hash-rate",
    "difficulty",
    "transaction-fees-usd",
    "n-unique-addresses",
    "n-transactions",
    "my-wallet-n-users",
    "utxo-count",
    "n-transactions-excluding-popular",
    "estimated-transaction-volume-usd",
    "trade-volume",
    "total-bitcoins",
    "market-price",
]

years = [2020]


class BitcoinPriceDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(BitcoinPriceDataLoader, self).__init__(config)

        print("Load data...")
        if config.trainer.with_sent:
            df = pd.read_pickle("./data/processed/chartdata_sent.pkl")
        else:
            df = pd.read_pickle("./data/processed/chartdata_sent.pkl")
            df.drop(["polarity"], axis=1, inplace=True)

        if config.trainer.test_size:
            test_size = config.trainer.test_size
        else:
            test_size = config.trainer.timestep+(config.trainer.prediction_length+1)

        print("Test Size: {}".format(test_size))

        df = self._create_technical_indicators(
            df,
            (config.trainer.model_task == "classification"),
            config.trainer.with_sent,
        )

        print(df.head())
        print(df.info())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df.iloc[:, :-1].to_numpy(),
            df.iloc[:, -1:].to_numpy(),
            test_size=test_size,
            shuffle=False,
            random_state=1,
        )


    def _create_technical_indicators(self, data, signal=False, with_sent=True):
        # data['SMA10'] = data['market-price'].rolling(
        #     window=10, min_periods=1, center=False).mean()
        # data['SMA50'] = data['market-price'].rolling(
        #     window=50, min_periods=1, center=False).mean()
        # data['SMA200'] = data['market-price'].rolling(
        #     window=200, min_periods=1, center=False).mean()
        # data['EMA10'] = data['market-price'].ewm(span=10).mean()
        # data['EMA50'] = data['market-price'].ewm(span=50).mean()
        # data['EMA200'] = data['market-price'].ewm(span=200).mean()
        if with_sent:
            data.loc[(data['polarity'] > .75), 'polarity'] = 2
            data.loc[(data['polarity'] <= .75), 'polarity'] = 1
            data.loc[(data['polarity'] <= .55), 'polarity'] = 0
            # data['PSMA10'] = data['polarity'].rolling(
            #     window=10, min_periods=1, center=False).mean()
            # data['PEMA10'] = data['polarity'].ewm(span=10).mean()
            # data[['polarity']] = data[['polarity']].astype(int)
        data["price"] = data["market-price"]
        data.drop(["market-price"], axis=1, inplace=True)
        # data["prev-price"] = data['price'].shift(1)
        # data['change'] = data[['prev-price', 'price']
        #                      ].pct_change(axis=1)['price']
        # data.dropna(subset=['change'], inplace=True)
        if signal:
            data[["next-price"]] = data[["price"]].shift(-1)
            data.loc[(data["price"] <= data["next-price"]), "signal"] = 1
            data.loc[(data["price"] >= data["next-price"]), "signal"] = 0
            data.dropna(subset=["signal"], inplace=True)
            data[["signal"]] = data[["signal"]].astype(int)
            data.drop(["next-price"], axis=1, inplace=True)
            # data.drop(['price'], axis=1, inplace=True)

        return data

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
