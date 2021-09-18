class BaseDataProducer(object):
    def __init__(self, config):
        self.config = config
        self.raw_data = self._retrieve_data()
        self.processed_data = self._preprocess_data()

    def _retrieve_data(self):
        raise NotImplementedError

    def _preprocess_data(self):
        raise NotImplementedError

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.processed_data