from base.base_trainer import BaseTrain
import os
from pathlib import Path
from joblib import load, dump

import numpy as np
import pandas as pd

# from plotly.offline import plot, iplot
# import plotly.graph_objs as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint  # , TensorBoard

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    classification_report,
    auc,
)
from sklearn.preprocessing import MinMaxScaler


class BitcoinPriceModelTrainer(BaseTrain):
    def __init__(self, model, train_data, test_data, config):
        super(BitcoinPriceModelTrainer, self).__init__(
            model, train_data, test_data, config
        )
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        # self.callbacks.append(
        #     ModelCheckpoint(
        #         filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
        #         monitor=self.config.callbacks.checkpoint_monitor,
        #         mode=self.config.callbacks.checkpoint_mode,
        #         save_best_only=self.config.callbacks.checkpoint_save_best_only,
        #         save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
        #         verbose=self.config.callbacks.checkpoint_verbose,
        #     )
        # )
        self.callbacks.append(
            ModelCheckpoint(
                self.config.trainer.model_name,
                monitor="val_loss",
                mode="min",
                verbose=1,
                save_best_only=True,
            )
        )
        # self.callbacks.append(
        #     TensorBoard(
        #         log_dir=self.config.callbacks.tensorboard_log_dir,
        #         write_graph=self.config.callbacks.tensorboard_write_graph,
        #     )
        # )

        # if hasattr(self.config,"comet_api_key"):
        #     from comet_ml import Experiment
        #     experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
        #     experiment.disable_mp()
        #     experiment.log_multiple_params(self.config)
        #     self.callbacks.append(experiment.get_keras_callback())

    def _scale_data(self, X_train, y_train, X_test, y_test):

        # print(y_test)
        train = np.concatenate((X_train, y_train), axis=1)
        # print(train[0])
        scaler = MinMaxScaler().fit(train)
        scaled_train = scaler.transform(train)
        # y_scaled_train = scaler.transform(y_train)
        test = np.concatenate((X_test, y_test), axis=1)
        scaled_test = scaler.transform(test)
        # y_scaled_test = scaler.transform(y_test)

        # print(scaled_train[0])
        print("Saving artifacts...")
        Path("artifacts").mkdir(exist_ok=True)
        dump(scaler, "artifacts/scaler.pkl")

        self.nscaler_features = scaled_train.shape[1]

        return (
            scaled_train[:, :-1],
            scaled_train[:, -1].reshape(-1, 1),
            scaled_test[:, :-1],
            scaled_test[:, -1].reshape(-1, 1),
        )

    def _inverse_data(self, X_test_pred, y_test):

        scaler = load("artifacts/scaler.pkl")
        y_test_dataset_like = np.zeros(shape=(len(y_test), self.nscaler_features))
        # put the predicted values in the right field
        y_test_dataset_like[:, -1] = y_test
        # inverse transform and then select the right field
        y_test_inverse = scaler.inverse_transform(y_test_dataset_like)[:, -1]

        trainPredict_dataset_like = np.zeros(
            shape=(len(X_test_pred), self.nscaler_features)
        )
        # put the predicted values in the right field
        trainPredict_dataset_like[:, -1] = X_test_pred[:, -1]
        # inverse transform and then select the right field
        X_test_pred_inverse = scaler.inverse_transform(trainPredict_dataset_like)[:, -1]

        return X_test_pred_inverse, y_test_inverse

    def _prepare_regression_data(self, X_data, y_data, look_back=5):

        dataX, dataY = [], []
        print(y_data.shape)
        for i in range(len(X_data) - look_back - 1):
            a = X_data[i : (i + look_back)]
            dataX.append(a)
            # print(i + look_back+1)
            # print(y_data[i + look_back+1, 0])
            dataY.append(y_data[i + look_back + 1, 0])

        return np.array(dataX), np.array(dataY)

    def train(self):

        X_scaled_train, y_scaled_train, X_scaled_test, y_scaled_test = self._scale_data(
            self.train_data[0], self.train_data[1], self.test_data[0], self.test_data[1]
        )
        self.X_train, self.y_train = self._prepare_regression_data(
            X_scaled_train, y_scaled_train, self.config.trainer.timestep
        )
        self.X_test, self.y_test = self._prepare_regression_data(
            X_scaled_test, y_scaled_test, self.config.trainer.timestep
        )
        print("************** Shape {}  {}".format(X_scaled_test.shape, y_scaled_test.shape))
        if self.config.trainer.model_type_current == "CNN":
            history = self.model.fit(
                self.X_train.reshape(
                    self.X_train.shape[0],
                    self.X_train.shape[1],
                    self.X_train.shape[2],
                    1,
                ),
                self.y_train,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_split=self.config.trainer.validation_split,
                shuffle=False,
                callbacks=self.callbacks,
            )
        elif self.config.trainer.model_type_current == "CNNLSTM":
            history = self.model.fit(
                self.X_train.reshape(
                    self.X_train.shape[0],
                    1,
                    self.X_train.shape[1],
                    self.X_train.shape[2],
                ),
                self.y_train,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_split=self.config.trainer.validation_split,
                shuffle=False,
                callbacks=self.callbacks,
            )
        elif self.config.trainer.model_type_current == "ConvLSTM2D_Encoder_Decoder":
            history = self.model.fit(
                self.X_train.reshape(
                    (
                        self.X_train.shape[0],
                        self.X_train.shape[1],
                        1,
                        1,
                        self.X_train.shape[2],
                    )
                ),
                self.y_train,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_split=self.config.trainer.validation_split,
                shuffle=False,
                callbacks=self.callbacks,
            )
        else:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                batch_size=self.config.trainer.batch_size,
                validation_split=self.config.trainer.validation_split,
                shuffle=False,
                callbacks=self.callbacks,
            )
        self.loss.extend(history.history["loss"])
        self.val_loss.extend(history.history["val_loss"])

    def evaluate(self):

        if self.config.trainer.model_task == "regression":
            if self.config.trainer.model_type_current == "CNN":
                X_test_pred = self.model.predict(
                    self.X_test.reshape(
                        self.X_test.shape[0],
                        self.X_test.shape[1],
                        self.X_test.shape[2],
                        1,
                    )
                )
            elif self.config.trainer.model_type_current == "CNNLSTM":
                X_test_pred = self.model.predict(
                    self.X_test.reshape(
                        self.X_test.shape[0],
                        1,
                        self.X_test.shape[1],
                        self.X_test.shape[2],
                    )
                )
            elif self.config.trainer.model_type_current == "ConvLSTM2D_Encoder_Decoder":
                X_test_pred = self.model.predict(
                    self.X_test.reshape(
                        (
                            self.X_test.shape[0],
                            self.X_test.shape[1],
                            1,
                            1,
                            self.X_test.shape[2],
                        )
                    )
                )
            else:
                X_test_pred = self.model.predict(self.X_test)

            X_test_pred_inverse, y_test_inverse = self._inverse_data(
                X_test_pred, self.y_test
            )

            # print("************** Ev Shape {}  {}".format(X_test_pred_inverse.shape, y_test_inverse.shape))
            # print(y_test_inverse)
            # mape = keras.losses.MAPE(y_test_inverse[:-5], X_test_pred_inverse[:-5])
            mape = keras.losses.MAPE(y_test_inverse, X_test_pred_inverse)
            # print(X_test_pred_inverse.shape)
            print("********************** MAPE")
            print(mape.numpy())

            if mape.numpy() <= 0.2:
                self.visualize_prediction(
                    y_test_inverse, X_test_pred_inverse, mape.numpy()
                )

            return_val = mape.numpy()
        else:
            if self.config.trainer.model_type_current == "CNN":
                X_test_pred = self.model.predict_classes(
                    self.X_test.reshape(
                        self.X_test.shape[0],
                        self.X_test.shape[1],
                        self.X_test.shape[2],
                        1,
                    )
                )
            elif self.config.trainer.model_type_current == "CNNLSTM":
                X_test_pred = self.model.predict_classes(
                    self.X_test.reshape(
                        self.X_test.shape[0],
                        1,
                        self.X_test.shape[1],
                        self.X_test.shape[2],
                    )
                )
            elif self.config.trainer.model_type_current == "ConvLSTM2D_Encoder_Decoder":
                X_test_pred = self.model.predict_classes(
                    self.X_test.reshape(
                        (
                            self.X_test.shape[0],
                            self.X_test.shape[1],
                            1,
                            1,
                            self.X_test.shape[2],
                        )
                    )
                )
            else:
                X_test_pred = self.model.predict_classes(self.X_test)

            accuracy = accuracy_score(self.y_test, X_test_pred)
            print("Accuracy ", accuracy)
            precision = precision_score(self.y_test, X_test_pred)
            print("Precision ", precision)
            recall = recall_score(self.y_test, X_test_pred)
            print("Recall ", recall)
            f1 = f1_score(self.y_test, X_test_pred)
            print("F1 ", f1)
            print("Predicted ", X_test_pred.flatten())
            print("Actual ", self.y_test.flatten())

            # if accuracy >= .75:
            #     self.visualize_prediction(self.y_test, X_test_pred)

            return_val = [accuracy, precision, recall, f1]

        return return_val
