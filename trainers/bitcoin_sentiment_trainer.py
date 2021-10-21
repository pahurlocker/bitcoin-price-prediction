from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint  # , TensorBoard
import numpy as np

# import matplotlib.pyplot as plt
# import plotly.express as px
# from plotly.offline import plot, iplot

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    classification_report,
    auc,
)


class BitcoinSentimentModelTrainer(BaseTrain):
    def __init__(self, model, train_data, test_data, config):
        super(BitcoinSentimentModelTrainer, self).__init__(
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

    def train(self):
        history = self.model.fit(
            self.train_data[0],
            self.train_data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["accuracy"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_accuracy"])

    def evaluate(self):
        scores = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1]*100))

        # X_test_pred = self.model.predict_classes(
        #     self.test_data[0], self.config.trainer.batch_size
        # )

        # X_test_pred = self.model.predict(
        #     self.test_data[0], self.config.trainer.batch_size
        # )
        # X_test_pred = np.argmax(X_test_pred,axis=1)
        X_test_pred = (self.model.predict(self.test_data[0], self.config.trainer.batch_size) > 0.5).astype("int32")

        accuracy = accuracy_score(self.test_data[1], X_test_pred)
        print("Accuracy ", accuracy)
        precision = precision_score(self.test_data[1], X_test_pred)
        print("Precision ", precision)
        recall = recall_score(self.test_data[1], X_test_pred)
        print("Recall ", recall)
        f1 = f1_score(self.test_data[1], X_test_pred)
        print("F1 ", f1)
        print(X_test_pred.flatten())
        print(self.test_data[1].flatten())
        return_val = [accuracy, precision, recall, f1]

        # self.visualize_prediction(self.test_data[1], X_test_pred)

        return return_val
