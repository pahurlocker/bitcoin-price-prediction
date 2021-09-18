from base.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from tcn import TCN
from attention import Attention

class BitcoinPriceLSTMClassificationModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceLSTMClassificationModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.LSTM(units=self.config.trainer.units, activation=self.config.trainer.activation_function,
                            batch_input_shape=(None, None, self.config.trainer.nfeatures)))
        self.model.add(layers.LeakyReLU(alpha=0.5))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                    metrics=['accuracy'])
        self.model.summary()

class BitcoinPriceCNNClassificationModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceCNNClassificationModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.Conv2D(filters=42, kernel_size=(3, self.config.trainer.nfeatures), input_shape=(
            self.config.trainer.timestep, self.config.trainer.nfeatures, 1), activation=self.config.trainer.activation_function))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                    metrics=['accuracy'])
        self.model.summary()

class BitcoinPriceEncoderDecoderConvLSTM2DClassificationModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceEncoderDecoderConvLSTM2DClassificationModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(layers.ConvLSTM2D(filters=64, padding="same", kernel_size=(
        1, 3), activation=self.config.trainer.activation_function, input_shape=(self.config.trainer.timestep, 1, 1, self.config.trainer.nfeatures)))
        self.model.add(layers.Flatten())
        self.model.add(layers.RepeatVector(1))
        self.model.add(layers.LSTM(
            self.config.trainer.units, activation=self.config.trainer.activation_function, return_sequences=True))
        # model.add(layers.TimeDistributed(layers.Dense(
        #     self.config.trainer.units, activation=self.config.trainer.activation_function)))
        # model.add(layers.TimeDistributed(layers.Dense(1)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                    metrics=['accuracy'])
        self.model.summary()

class BitcoinPriceTCNClassificationModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceTCNClassificationModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(None, self.config.trainer.nfeatures)))
        self.model.add(TCN(nb_filters=self.config.trainer.units, kernel_size=3,
                    dilations=self.config.trainer.dilations, padding='same', dropout_rate=.05))
        # model.add(TCN(nb_filters=self.config.trainer.units, kernel_size=3,
        #              dilations=dilations, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                    metrics=['accuracy'])
        self.model.summary()

class BitcoinPriceAttentionClassificationModel(BaseModel):
    def __init__(self, optimizer, config):
        super(BitcoinPriceAttentionClassificationModel, self).__init__(optimizer, config)
        self.build_model()

    def build_model(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.config.trainer.timestep, self.config.trainer.nfeatures)))
        self.model.add(tf.keras.layers.LSTM(units=self.config.trainer.units, return_sequences=True))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        # model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(Attention(self.config.trainer.units/2))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                    metrics=['accuracy'])
        self.model.summary()