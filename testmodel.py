from model import Model
import numpy as np

from max_pooling import MaxPoolLayer
from convolution import ConvolutionLayer
from fully_connected import FullyConnectedLayer
from flattening import FlatteningLayer
from relu_activation import ReLU
from softmax import Softmax
from evaluation_metrices import accuracy


import pickle


class TestNet(Model):
    def __init__(self, alpha=0.001, initializer='xavier_uniform', num_classes=10, filename='testnet.pickle'):
        self.layers = []
        self.params = []
        self.alpha = alpha
        self.initializer = initializer
        self.num_classes = num_classes
        self.filename = filename

        self._build_model()
        pass

    def _build_model(self):
        
        # convolution layer ========================================
        params = {
            'filter_dim'    :    (6, 6),
            'stride'        :    (2, 2),
            'padding'       :    (0, 0),
            'ch_out'        :    16,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }
        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)


        # max pooling layer ========================================
        params={
            'filter_dim': (3, 3), 
            'stride': (2, 2)
        }
        self.layers.append(MaxPoolLayer(params=params))
        self.params.append(params)


        # flattening layer ===========================================
        self.layers.append(FlatteningLayer())
        self.params.append(None)


        # fully connected layer 1 ====================================
        params = {
            'a_out'         :    100,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(FullyConnectedLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)
        
        # fully connected layer 2 ====================================
        params = {
            'a_out'         :    self.num_classes,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(FullyConnectedLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # softmax layer ==============================================
        self.layers.append(Softmax())
        self.params.append(None)

        pass



    def train(self, train_data):
        X, y = train_data

        # forward prop
        for layer in self.layers:
            X = layer.forward_prop(X)
            
        # calc loss
        Z = X - y
        loss = self.calc_loss(y, X)

        # backward prop
        for layer in reversed(self.layers):
            Z = layer.backward_prop(Z)['dx']

        return loss


    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_prop(X)
        return X

    def save(self):
        
        with open(self.filename, 'wb') as file:
            pickle.dump(self.layers, file)


    def setLayers(self, layers):
        self.layers = layers

    
    def evaluate(self, data):
        self.accuracy = accuracy(self.predict(data[0]), data[1])
        return self.accuracy

    
    def calc_loss(self, y, y_hat):
        """
        Calculate the loss

        Arguments:
        y -- numpy array of shape (batch_size, n_classes)
        y_hat -- numpy array of shape (batch_size, n_classes)
        
        """

        loss = -np.sum(y * np.log(y_hat + 1e-8)) / y.shape[0]

        return loss