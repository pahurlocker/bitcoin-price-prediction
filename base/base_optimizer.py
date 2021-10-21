import os
import glob


class BaseOptimizer(object):
    def __init__(self, config):
        # self.train_data = train_data
        # self.test_data = test_data
        self.config = config

    def clean_models(self, study, model_name_prefix):
        # Rename best model
        best_model_name = "{}-tmp-{}.h5".format(
            model_name_prefix, study.best_trial.number
        )
        final_name = model_name_prefix + "-final.h5"
        os.rename(best_model_name, final_name)
        # Remove the other models
        rm_mdls = glob.glob(model_name_prefix + "-tmp-*")
        for mdl in rm_mdls:
            os.remove(mdl)

    def optimize(self):
        raise NotImplementedError
