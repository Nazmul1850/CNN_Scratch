from abc import ABC, abstractmethod


LEARNING_RATE = 0.001
INITIALIZER = 'xaiver_uniform'
NUM_CLASSES = 10

class Model(ABC):
    @abstractmethod
    def __init__(self, alpha, initializer, num_classes, filename):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        pass
    
    def evaluate(self, data):
        pass
    
    def predict(self, data):
        pass

    def cald_loss(self, y, y_hat):
        pass