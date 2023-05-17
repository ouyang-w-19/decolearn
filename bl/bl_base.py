from abc import ABC, abstractmethod


class BLBase(ABC):
    """ Base class of base learner"""
    def __init__(self, input_dims, lr, loss, id, inverse_factor=1):
        self._input_dims = input_dims
        self._lr = lr
        self._id = id
        self._loss = loss
        self.model = self.build_model()
        self.invert_factor = inverse_factor
        self.accuracy = -1

    def invert_output(self, do_invert: bool):
        if not type(do_invert) is bool:
            raise ValueError('Error: Parameter "inverse_output" is a boolean!')
        if do_invert:
            self.invert_factor *= -1

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, x, y, batch_size, epochs, callbacks=None):
        pass

    @abstractmethod
    def get_prediction(self, inp, batch_size, binary):
        pass

    @abstractmethod
    def _retrieve_model_info(self):
        pass

