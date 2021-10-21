from base.base_optimizer import BaseOptimizer
from models.bitcoin_price_regression_models import *
from data_loaders.bitcoin_price_dataloader import BitcoinPriceDataLoader
from trainers.bitcoin_price_trainer import BitcoinPriceModelTrainer
from keras.backend import clear_session

from rubicon_ml import Rubicon
from rubicon_ml.ui import Dashboard

import optuna
import joblib

import datetime

root_dir = "./artifacts/rubicon-root"
rubicon = Rubicon(persistence="filesystem", root_dir=root_dir)


class BitcoinPriceRegressionOptimizer(BaseOptimizer):
    def __init__(self, config):
        super(BitcoinPriceRegressionOptimizer, self).__init__(
            config
        )
        self.project = rubicon.get_or_create_project(
            "Bitcoin Price Prediction (Regression v1)", description=""
        )

        self.study = optuna.create_study(storage="sqlite:///artifacts/db.sqlite3", 
                study_name="bitcoin-regression-{}-{}".format(self.config.optimizer.version,
                datetime.datetime.now().strftime("%Y-%m-%d%H%M%S%z")), direction="minimize")

    def _objective(self, trial):
        return_val = 0

        clear_session()

        data_loader = BitcoinPriceDataLoader(self.config)
        self.config.trainer.nfeatures = int(data_loader.get_train_data()[0].shape[1])

        if self.config.trainer.model_type == "":
            self.config.trainer.model_type_current = trial.suggest_categorical(
                "model_type",
                [
                    "LSTM",
                    "CNN",
                    "GRU",
                    "CNNLSTM",
                    "Attention",
                    "LSTM_Encoder_Decoder",
                    "TCN",
                    "ConvLSTM2D_Encoder_Decoder",
                ],
            )
        else:
            self.config.trainer.model_type_current = self.config.trainer.model_type

        if self.config.trainer.with_sent:
            sent_tag = "with-sent"
        else:
            sent_tag = "without-sent"

        experiment = self.project.log_experiment(
            name=trial.number,
            model_name="Bitcoin Price Regression",
            tags=[self.config.trainer.model_type_current, sent_tag, "without-ind"],
        )

        self.config.trainer.verbose_training = 0
        self.config.trainer.validation_split = 0.1

        if self.config.trainer.timestep_set == False:
            self.config.trainer.timestep = trial.suggest_categorical(
                "timesteps", [5, 10, 20, 30]
            )
        else:
            self.config.trainer.timestep = trial.suggest_categorical(
                "timesteps", [self.config.trainer.timestep]
            )
        experiment.log_parameter("timestep", self.config.trainer.timestep)

        self.config.trainer.batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32]
        )  # , 16, 32, 64])
        experiment.log_parameter("batch_size", self.config.trainer.batch_size)

        experiment.log_parameter("model_type", self.config.trainer.model_type_current)

        self.config.trainer.num_epochs = trial.suggest_categorical(
            "epochs", [10, 20, 50, 100]
        )
        experiment.log_parameter("epochs", self.config.trainer.num_epochs)

        self.config.trainer.activation_function = trial.suggest_categorical(
            "activation_function", ["tanh", "relu"]
        )
        experiment.log_parameter("activation", self.config.trainer.activation_function)

        self.config.trainer.optimizer_name = trial.suggest_categorical(
            "optimizer", ["adam"]
        )
        experiment.log_parameter("optimizer", self.config.trainer.optimizer_name)

        if self.config.trainer.optimizer_name == "adam":
            adam_lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
            experiment.log_parameter("adam_lr", adam_lr)
            optimizer = keras.optimizers.Adam(lr=adam_lr)
        else:
            sgd_lr = trial.suggest_loguniform("sgd_lr", 1e-5, 1e-1)
            experiment.log_parameter("sgd_lr", sgd_lr)
            sgd_momentum = trial.suggest_loguniform("sgd_momentum", 1e-5, 1e-1)
            experiment.log_parameter("sgd_momentum", sgd_momentum)
            sgd_nesterov = trial.suggest_categorical("sgd_nesterov", [False, True])
            experiment.log_parameter("sgd_nesterov", sgd_nesterov)
            optimizer = keras.optimizers.SGD(
                lr=sgd_lr, momentum=sgd_momentum, nesterov=sgd_nesterov, clipvalue=0.5
            )

        self.config.trainer.model_name = "{}-tmp-{}.h5".format(
            self.config.trainer.model_name_prefix, trial.number
        )

        print("Create the model.")
        if self.config.trainer.model_type_current == "LSTM":
            self.config.trainer.units = trial.suggest_categorical("units", [16, 32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceLSTMRegressionModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == "GRU":
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceGRURegressionModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == "CNN":
            model = BitcoinPriceCNNRegressionModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == "CNNLSTM":
            self.config.trainer.filters = trial.suggest_categorical("filters", [40, 50])
            experiment.log_parameter("filters", self.config.trainer.filters)
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceCNNLSTMRegressionModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == "ConvLSTM2D_Encoder_Decoder":
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceEncoderDecoderConvLSTM2DRegressionModel(
                optimizer, self.config
            )
        elif self.config.trainer.model_type_current == "LSTM_Encoder_Decoder":
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceEncoderDecoderLSTMRegressionModel(
                optimizer, self.config
            )
        # elif self.config.trainer.model_type_current == "TCN":
        #     self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
        #     experiment.log_parameter("units", self.config.trainer.units)
        #     self.config.trainer.dilations = trial.suggest_categorical(
        #         "dilations",
        #         [
        #             (1, 2, 4, 8),
        #             (1, 2, 4, 8, 16),
        #             (1, 2, 4, 8, 16, 32),
        #             (1, 2, 4, 8, 16, 32, 64),
        #         ],
        #     )
        #     model = BitcoinPriceTCNRegressionModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == "Attention":
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceAttentionRegressionModel(optimizer, self.config)

        print("Create the trainer")
        trainer = BitcoinPriceModelTrainer(
            model.model, data_loader.get_train_data(), data_loader.get_test_data(), self.config
        )

        print("Start training the model.")
        trainer.train()

        print("Evaluate model performance")
        return_val = trainer.evaluate()
        experiment.log_metric("MAPE", return_val)

        return return_val

    def optimize(self):
        self.study.optimize(
            lambda trial: self._objective(trial), self.config.optimizer.n_trials
        )  # , timeout=1200)

        print("Number of finished trials: {}".format(len(self.study.trials)))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        joblib.dump(
            self.study,
            "./artifacts/study-bitcoin-prediction-d-{}-v{}.pkl".format(
                self.config.optimizer.version,
                datetime.datetime.now().strftime("%Y-%m-%d%H%M%S%z"),
            ),
        )

        self.clean_models(self.study, self.config.trainer.model_name_prefix)
        # Dashboard(persistence="filesystem", root_dir=root_dir).run_server()

