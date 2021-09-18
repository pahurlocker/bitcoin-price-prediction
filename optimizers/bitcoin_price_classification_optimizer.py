from base.base_optimizer import BaseOptimizer
from models.bitcoin_price_classification_models import *
from trainers.bitcoin_price_trainer import BitcoinPriceModelTrainer
from keras.backend import clear_session

from rubicon import Rubicon
from rubicon.ui import Dashboard

import optuna
import joblib

import datetime

root_dir = "./artifacts/rubicon-root"
rubicon = Rubicon(persistence="filesystem", root_dir=root_dir)

class BitcoinPriceClassificationOptimizer(BaseOptimizer):
    def __init__(self, train_data, test_data, config):
        super(BitcoinPriceClassificationOptimizer, self).__init__(train_data, test_data, config)
        self.project = rubicon.get_or_create_project(
            "Bitcoin Price Prediction (Classification v1)", description="")

        self.study = optuna.create_study(direction="maximize")
    
    def _objective(self, trial):
        return_val = 0

        clear_session()

        if self.config.trainer.model_type == '':
            self.config.trainer.model_type_current = trial.suggest_categorical(
                "model_type", ['LSTM', 'CNN', 'Attention', 'TCN', 'ConvLSTM2D_Encoder_Decoder']) 
        else:
            self.config.trainer.model_type_current = self.config.trainer.model_type

        if self.config.trainer.with_sent:
            sent_tag = 'with-sent'
        else:
            sent_tag = 'without-sent'

        experiment = self.project.log_experiment(
            name=trial.number,
            model_name="Bitcoin Price Classification",
            tags=[self.config.trainer.model_type_current, sent_tag,'without-ind'],
        )

        self.config.trainer.verbose_training = 1
        self.config.trainer.validation_split = .1

        self.config.trainer.timestep = trial.suggest_categorical('timesteps', [5, 10, 20])
        experiment.log_parameter("timestep", self.config.trainer.timestep)

        self.config.trainer.batch_size = trial.suggest_categorical("batch_size", [8, 16])
        experiment.log_parameter("batch_size", self.config.trainer.batch_size)

        experiment.log_parameter("model_type", self.config.trainer.model_type_current)

        self.config.trainer.num_epochs = trial.suggest_categorical(
            "epochs", [10, 20, 50, 100])
        experiment.log_parameter("epochs", self.config.trainer.num_epochs)

        self.config.trainer.activation_function = trial.suggest_categorical(
            "activation_function", ['tanh', 'relu'])
        experiment.log_parameter("activation", self.config.trainer.activation_function)

        self.config.trainer.optimizer_name = trial.suggest_categorical('optimizer', ['adam'])
        experiment.log_parameter("optimizer", self.config.trainer.optimizer_name)

        if self.config.trainer.optimizer_name == 'adam':
            adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
            experiment.log_parameter("adam_lr", adam_lr)
            optimizer = keras.optimizers.Adam(lr=adam_lr)
        else:
            sgd_lr = trial.suggest_loguniform('sgd_lr', 1e-5, 1e-1)
            experiment.log_parameter("sgd_lr", sgd_lr)
            sgd_momentum = trial.suggest_loguniform('sgd_momentum', 1e-5, 1e-1)
            experiment.log_parameter("sgd_momentum", sgd_momentum)
            sgd_nesterov = trial.suggest_categorical('sgd_nesterov', [False, True])
            experiment.log_parameter("sgd_nesterov", sgd_nesterov)
            optimizer = keras.optimizers.SGD(
                lr=sgd_lr, momentum=sgd_momentum, nesterov=sgd_nesterov, clipvalue=0.5)

        self.config.trainer.model_name = "{}-tmp-{}.h5".format(self.config.trainer.model_name_prefix, trial.number)

        print('Create the model.')
        if self.config.trainer.model_type_current == 'LSTM':
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceLSTMClassificationModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == 'CNN':
            model = BitcoinPriceCNNClassificationModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == 'ConvLSTM2D_Encoder_Decoder':
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceEncoderDecoderConvLSTM2DClassificationModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == 'TCN':
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            self.config.trainer.dilations = trial.suggest_categorical('dilations', [(1, 2, 4, 8), (1, 2, 4, 8, 16),
                                                                (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)])
            model = BitcoinPriceTCNClassificationModel(optimizer, self.config)
        elif self.config.trainer.model_type_current == 'Attention':
            self.config.trainer.units = trial.suggest_categorical("units", [32, 64])
            experiment.log_parameter("units", self.config.trainer.units)
            model = BitcoinPriceAttentionClassificationModel(optimizer, self.config)

        print('Create the trainer')
        trainer = BitcoinPriceModelTrainer(model.model, self.train_data, self.test_data, self.config)

        print('Start training the model.')
        trainer.train()

        print('Evaluate model performance')
        return_val = trainer.evaluate()
        experiment.log_metric("Accuracy", return_val[0])
        experiment.log_metric("Precision", return_val[1])
        experiment.log_metric("Recall", return_val[2])
        experiment.log_metric("F1", return_val[3])

        return return_val[0]

    def optimize(self):
        self.study.optimize(lambda trial: self._objective(trial),
                   self.config.optimizer.n_trials)  # , timeout=1200)

        print("Number of finished trials: {}".format(len(self.study.trials)))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        joblib.dump(self.study, "./artifacts/study-bitcoin-prediction-d-{}-v{}.pkl".format(self.config.optimizer.version,
                                                                                    datetime.datetime.now().strftime('%Y-%m-%d%H%M%S%z')))

        self.clean_models(self.study, self.config.trainer.model_name_prefix)
        Dashboard(persistence="filesystem",
                root_dir=root_dir).run_server()
        