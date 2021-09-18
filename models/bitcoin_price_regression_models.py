from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from tcn import TCN
from attention import Attention

class BitcoinPriceLSTMRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceLSTMRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.LSTM(units=self.config.trainer.units, activation=self.config.trainer.activation_function,
                            batch_input_shape=(None, None, self.config.trainer.nfeatures)))
        self.model.add(layers.LeakyReLU(alpha=0.5))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceGRURegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceGRURegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.GRU(units=self.config.trainer.units, activation=self.config.trainer.activation_function,
                            batch_input_shape=(None, None, self.config.trainer.nfeatures)))
        self.model.add(layers.LeakyReLU(alpha=0.5))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceCNNRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceCNNRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.Conv2D(filters=40, kernel_size=(3, self.config.trainer.nfeatures), input_shape=(
            self.config.trainer.timestep, self.config.trainer.nfeatures, 1), activation=self.config.trainer.activation_function))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceCNNLSTMRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceCNNLSTMRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.TimeDistributed(layers.Conv1D(filters=self.config.trainer.filters, kernel_size=1,
                                                    activation=self.config.trainer.activation_function), input_shape=(None, self.config.trainer.timestep, self.config.trainer.nfeatures)))
        self.model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.LSTM(self.config.trainer.units, activation=self.config.trainer.activation_function))
        self.model.add(layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceEncoderDecoderConvLSTM2DRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceEncoderDecoderConvLSTM2DRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.ConvLSTM2D(filters=64, padding="same", kernel_size=(
            1, 3), activation=self.config.trainer.activation_function, input_shape=(self.config.trainer.timestep, 1, 1, self.config.trainer.nfeatures), return_sequences=True))
        self.model.add(layers.MaxPooling3D(pool_size=(1, 1, 1)))
        self.model.add(layers.Flatten())
        self.model.add(layers.RepeatVector(1))
        self.model.add(layers.LSTM(
            self.config.trainer.units, activation=self.config.trainer.activation_function, return_sequences=True))
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.TimeDistributed(layers.Dense(
            self.config.trainer.units, activation=self.config.trainer.activation_function)))
        self.model.add(layers.TimeDistributed(layers.Dense(1)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceEncoderDecoderLSTMRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceEncoderDecoderLSTMRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.LSTM(self.config.trainer.units, activation=self.config.trainer.activation_function,
                            input_shape=(None, self.config.trainer.nfeatures)))
        self.model.add(layers.RepeatVector(self.config.trainer.timestep))
        self.model.add(layers.LSTM(
            self.config.trainer.units, activation=self.config.trainer.activation_function, return_sequences=True))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceTCNRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceTCNRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(None, self.config.trainer.nfeatures)))
        self.model.add(TCN(nb_filters=self.config.trainer.units, kernel_size=3,
                    dilations=self.config.trainer.dilations, padding='same', dropout_rate=.05))
        # self.model.add(TCN(nb_filters=units, kernel_size=3,
        #              dilations=dilations, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()

class BitcoinPriceAttentionRegressionModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceAttentionRegressionModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.config.trainer.timestep, self.config.trainer.nfeatures)))
        self.model.add(tf.keras.layers.LSTM(units=self.config.trainer.units, return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(Attention(self.config.trainer.units/2))
        self.model.add(tf.keras.layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.model.summary()
