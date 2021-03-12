import sys
import getopt
import yaml

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from keras.backend import clear_session
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.constraints import max_norm

import optuna
import joblib

from plotly.offline import plot, iplot
import plotly.graph_objs as go

import datetime

metrics = ['blocks-size', 'avg-block-size', 'n-transactions-total',
           'hash-rate', 'difficulty', 'transaction-fees-usd',
           'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
           'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
           'trade-volume', 'total-bitcoins', 'market-price']

years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]


def retrieve_data(metrics, years, pull_data):
    df_all = pd.DataFrame()

    if pull_data:
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

        df_all.to_pickle('chartdata.pkl')
    else:
        df_all = pd.read_pickle('./chartdata.pkl')

    return df_all


def get_config():
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def clean_data(data):
    data.dropna(subset=['market-price'], inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)

    return data


def create_technical_indicators(data, signal=False):
    data['SMA10'] = data['market-price'].rolling(
        window=10, min_periods=1, center=False).mean()
    data['SMA50'] = data['market-price'].rolling(
        window=50, min_periods=1, center=False).mean()
    data['SMA200'] = data['market-price'].rolling(
        window=200, min_periods=1, center=False).mean()
    data['EMA10'] = data['market-price'].ewm(span=10).mean()
    data['EMA50'] = data['market-price'].ewm(span=50).mean()
    data['EMA200'] = data['market-price'].ewm(span=200).mean()
    data['price'] = data['market-price']
    data.drop(['market-price'], axis=1, inplace=True)
    if (signal):
        data[['next-price']] = data[['price']].shift(-1)
        data.loc[(data['price'] <= data['next-price']), 'signal'] = 1
        data.loc[(data['price'] >= data['next-price']), 'signal'] = 0
        data.dropna(subset=['signal'], inplace=True)
        data[['signal']] = data[['signal']].astype(int)
        data.drop(['next-price'], axis=1, inplace=True)

    return data


def create_signal(data):
    data[['next-price']] = data[['market-price']].shift(-1)
    data.loc[(data['market-price'] <= data['next-price']), 'signal'] = 1
    data.loc[(data['market-price'] >= data['next-price']), 'signal'] = 0
    data.dropna(subset=['signal'], inplace=True)
    data[['signal']] = data[['signal']].astype(int)

    return data


def split_data(data, percentage=0.8):
    training_start = int(len(data) * percentage)

    train = data.iloc[:training_start]
    test = data.iloc[training_start:]
    print("Total datasets' length: ", train.shape, test.shape)

    return train, test


def scale_data(train, test, scaler):
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    return scaled_train, scaled_test


def inverse_data(X_test_pred, y_test, scaled_train, scaled_test, scaler):
    y_test_dataset_like = np.zeros(shape=(len(y_test), scaled_test.shape[1]))
    # put the predicted values in the right field
    y_test_dataset_like[:, -1] = y_test
    # inverse transform and then select the right field
    y_test_inverse = scaler.inverse_transform(y_test_dataset_like)[:, -1]

    trainPredict_dataset_like = np.zeros(
        shape=(len(X_test_pred), scaled_train.shape[1]))
    # put the predicted values in the right field
    trainPredict_dataset_like[:, -1] = X_test_pred[:, -1]
    # inverse transform and then select the right field
    X_test_pred_inverse = scaler.inverse_transform(
        trainPredict_dataset_like)[:, -1]

    return X_test_pred_inverse, y_test_inverse


def prepare_regression_data(data, look_back=5):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        dataX.append(a[:, :-1])  # Remove target value
        dataY.append(data[i + look_back+1, data.shape[1]-1])

    return np.array(dataX), np.array(dataY)


def prepare_classification_data(data, look_back=5):
    dataX, dataY = [], []
    for i in range(len(data)-look_back):
        a = data[i:(i+look_back)]
        dataX.append(a[:, :-1])  # Remove target value
        dataY.append(data[i + look_back-1, data.shape[1]-1])

    return np.array(dataX), np.array(dataY).astype(int)


def regression_model_LSTM(tsteps, nfeatures, units=32, learning_rate=0.001, activation_function='relu', loss_function='mse'):
    adam = keras.optimizers.Adam(lr=learning_rate)

    model = Sequential()
    model.add(layers.LSTM(units=units, activation=activation_function,
                          batch_input_shape=(None, tsteps, nfeatures), activity_regularizer=keras.regularizers.l1(0.001)))
    model.add(layers.LeakyReLU(alpha=0.5))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1, activation="linear"))

    model.compile(optimizer=adam, loss=loss_function)
    model.summary()

    return model


def regression_model_CNN(tsteps, nfeatures, learning_rate=0.001, activation_function='relu', loss_function='mse'):
    adam = keras.optimizers.Adam(lr=learning_rate)

    model = Sequential()
    model.add(layers.Conv2D(filters=40, kernel_size=(3, 20), input_shape=(
        tsteps, nfeatures, 1), activation=activation_function))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    model.compile(optimizer=adam, loss=loss_function)
    model.summary()

    return model


def classification_model_LSTM(tsteps, nfeatures, units=32, learning_rate=0.001, activation_function='relu', loss_function='binary_crossentropy'):
    adam = keras.optimizers.Adam(lr=learning_rate)
    sgd = keras.optimizers.SGD(lr=learning_rate, clipvalue=0.5)

    model = Sequential()
    model.add(layers.LSTM(units=units, activation=activation_function,
                          batch_input_shape=(None, tsteps, nfeatures)))
    model.add(layers.LeakyReLU(alpha=0.5))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model


def classification_model_CNN(tsteps, nfeatures, learning_rate=0.001, activation_function='tanh', loss_function='binary_crossentropy'):
    adam = keras.optimizers.Adam(lr=learning_rate)
    sgd = keras.optimizers.SGD(lr=learning_rate, clipvalue=0.5)

    model = Sequential()
    model.add(layers.Conv2D(filters=42, kernel_size=(3, 21), input_shape=(
        tsteps, nfeatures, 1), activation=activation_function))
    model.add(layers.Flatten())

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])
    model.summary()

    return model


def run_model(X_train, y_train, model, batch_size=64, epochs=200, validation_split=0.1, shuffle=False, checkpoint_name='best_model.h5'):
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = keras.callbacks.ModelCheckpoint(
        checkpoint_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_split=validation_split, shuffle=shuffle, callbacks=[es, mc])

    return model, history


def visualize_train_test_split(train, test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train['price'], name='Train Price'))
    fig.add_trace(go.Scatter(x=test.index,
                             y=test['price'], name='Test Price'))
    fig.update_layout(title='Historical Bitcoin Price Train & Test Split',
                      showlegend=True,
                      yaxis=dict(
                          title='BTC Price USD'))

    iplot(fig, filename="Historical Bitcoin Price Train & Test Split")


def visualize_prediction(y_test_inverse, X_test_pred_inverse, trial_number):
    trace1 = go.Scatter(
        x=np.arange(len(y_test_inverse)),
        y=y_test_inverse,
        mode='lines',
        name='Original Price'
    )
    trace2 = go.Scatter(
        x=np.arange(len(X_test_pred_inverse)),
        y=X_test_pred_inverse,
        mode='lines',
        name='Predicted Price'
    )
    layout = dict(
        title='Original vs Predicted Values Curve '+str(trial_number),
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type='linear'
        ),
        yaxis=dict(title='BTC Price USD')
    )
    pdata = [trace1, trace2]
    fig = dict(data=pdata, layout=layout)
    iplot(fig, filename="Time Series with Rangeslider")


def objective(trial, train, test):
    config = get_config()
    return_val = 0

    scaler = MinMaxScaler().fit(train)
    scaled_train, scaled_test = scale_data(train, test, scaler)

    clear_session()

    timestep = trial.suggest_categorical('timesteps', [5, 10, 20, 50])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    if config['model_problem'] == 'regression':
        X_train, y_train = prepare_regression_data(scaled_train, timestep)
        X_test, y_test = prepare_regression_data(scaled_test, timestep)

        tsteps = X_train.shape[1]
        nfeatures = X_train.shape[2]

        if config['model_type'] == 'LSTM':
            model_name = "model_pred-reg-lstm_v{}-{}.h5".format(
                datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z'), trial.number)

            model = regression_model_LSTM(tsteps, nfeatures, trial.suggest_categorical("units", [32, 64]),
                                          trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True), trial.suggest_categorical("activation_function", ['tanh', 'relu', 'sigmoid']), 'mse')

            model, history = run_model(X_train, y_train, model, batch_size, trial.suggest_categorical("epochs", [10, 20, 50, 100, 200]),
                                       config['validation_split'], config['shuffle'], model_name)

            # X_test_pred = model.predict(X_test)
            saved_model = load_model(model_name)
            X_test_pred = saved_model.predict(X_test)
        elif config['model_type'] == 'CNN':
            model_name = "model_pred-reg-cnn_v{}-{}.h5".format(
                datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z'), trial.number)

            model = regression_model_CNN(tsteps, nfeatures, trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                                         trial.suggest_categorical("activation_function", ['tanh', 'relu', 'sigmoid']), 'mse')

            model, history = run_model(X_train.reshape(X_train.shape[0], tsteps, nfeatures, 1), y_train, model,
                                       batch_size, trial.suggest_categorical(
                "epochs", [10, 20, 50, 100, 200]), config['validation_split'],
                config['shuffle'], model_name)

            # X_test_pred = model.predict(X_test.reshape(X_test.shape[0],tsteps,nfeatures,1))
            saved_model = load_model(model_name)
            X_test_pred = saved_model.predict(
                X_test.reshape(X_test.shape[0], tsteps, nfeatures, 1))

        X_test_pred_inverse, y_test_inverse = inverse_data(
            X_test_pred, y_test, scaled_train, scaled_test, scaler)

        mape = keras.losses.MAPE(
            y_test_inverse, X_test_pred_inverse
        )
        return_val = mape.numpy()
        print('Mean Absolute Percentage Error ', str(return_val))
        visualize_prediction(y_test_inverse, X_test_pred_inverse, trial.number)
    elif config['model_problem'] == 'classification':
        timestep = trial.suggest_categorical('timesteps', [5, 10, 20, 50])
        X_train, y_train = prepare_classification_data(scaled_train, timestep)
        X_test, y_test = prepare_classification_data(scaled_test, timestep)

        tsteps = X_train.shape[1]
        nfeatures = X_train.shape[2]

        if config['model_type'] == 'LSTM':
            model_name = "model_pred-cls-lstm_v{}-{}.h5".format(
                datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z'), trial.number)

            model = classification_model_LSTM(tsteps, nfeatures, trial.suggest_categorical("units", [32, 64]), trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                                              trial.suggest_categorical("activation_function", ['tanh', 'relu', 'sigmoid']), 'binary_crossentropy')

            model, history = run_model(X_train, y_train, model, batch_size, trial.suggest_categorical("epochs", [10, 20, 50, 100, 200]),
                                       config['validation_split'], config['shuffle'], model_name)

            #y_hat = model.predict_classes(X_test, batch_size)
            saved_model = load_model(model_name)
            y_hat = saved_model.predict_classes(X_test, batch_size)
            print(y_hat.flatten())
        elif config['model_type'] == 'CNN':
            model_name = "model_pred-cls-cnn_v{}-{}.h5".format(
                datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z'), trial.number)

            model = classification_model_CNN(tsteps, nfeatures, trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                                             trial.suggest_categorical("activation_function", ['tanh', 'relu', 'sigmoid']), 'binary_crossentropy')

            model, history = run_model(X_train.reshape(X_train.shape[0], tsteps, nfeatures, 1), y_train, model,
                                       batch_size, trial.suggest_categorical(
                "epochs", [10, 20, 50, 100, 200]), config['validation_split'],
                config['shuffle'], model_name)

            # y_hat = model.predict_classes(X_test.reshape(X_test.shape[0], tsteps, nfeatures, 1),batch_size)
            saved_model = load_model(model_name)
            y_hat = saved_model.predict_classes(X_test.reshape(
                X_test.shape[0], tsteps, nfeatures, 1), batch_size)
            print(y_hat.flatten())

        return_val = accuracy_score(y_test, y_hat)
        print("Accuracy ", accuracy_score(y_test, y_hat))

    return return_val


def main():

    config = get_config()

    df_all = retrieve_data(metrics, years, config['pull_data'])
    df_all.head()
    df_all = clean_data(df_all)
    df_all.head()
    df_all = create_technical_indicators(
        df_all, (config['model_problem'] == 'classification'))
    train, test = split_data(df_all)
    visualize_train_test_split(train, test)

    if config['model_problem'] == 'regression':
        study = optuna.create_study(direction="minimize")
    else:
        study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train, test),
                   config['n_trials'])  # , timeout=1200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    joblib.dump(study, "study_lstm_v{}.pkl".format(
        datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z')))


if __name__ == "__main__":
    main()
