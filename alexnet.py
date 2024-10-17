from model import Model

from max_pooling import MaxPoolLayer
from convolution import ConvolutionLayer
from fully_connected import FullyConnectedLayer
from flattening import FlatteningLayer
from relu_activation import ReLU
from softmax import Softmax

from evaluation_metrices import accuracy

import pickle

class AlexNet(Model):
    def __init__(self, alpha=0.001, initializer='xavier_uniform', num_classes=10, filename='alexnet.pickle'):
        self.layers = []
        self.params = []
        self.alpha = alpha
        self.initializer = initializer
        self.num_classes = num_classes
        self.filename = filename

        self._build_model()
        pass

    def _build_model(self):
        
        # convolution layer 1 ========================================
        params = {
            'filter_dim'    :    (11, 11),
            'stride'        :    (4, 4),
            'padding'       :    (0, 0),
            'ch_out'        :    96,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }
        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)


        # max pooling layer 1 ========================================
        params={
            'filter_dim': (3, 3), 
            'stride': (2, 2)
        }
        self.layers.append(MaxPoolLayer(params=params))
        self.params.append(params)


        # convolution layer 2 ========================================
        params = {
            'filter_dim'    :    (5, 5),
            'stride'        :    (1, 1),
            'padding'       :    (2, 2),
            'ch_out'        :    256,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }
        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # max pooling layer 2 ========================================
        params={
            'filter_dim': (3, 3),
            'stride': (2, 2)
        }

        self.layers.append(MaxPoolLayer(params=params))
        self.params.append(params)


        # convolution layer 3 ========================================
        params = {
            'filter_dim'    :    (3, 3),
            'stride'        :    (1, 1),
            'padding'       :    (1, 1),
            'ch_out'        :    384,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }
        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # convolution layer 4 ========================================
        params = {
            'filter_dim'    :    (3, 3),
            'stride'        :    (1, 1),
            'padding'       :    (1, 1),
            'ch_out'        :    384,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # convolution layer 5 ========================================
        params = {
            'filter_dim'    :    (3, 3),
            'stride'        :    (1, 1),
            'padding'       :    (1, 1),
            'ch_out'        :    256,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(ConvolutionLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # max pooling layer 3 ========================================
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
            'a_out'         :    4096,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(FullyConnectedLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # fully connected layer 2 ====================================
        params = {
            'a_out'         :    4096,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(FullyConnectedLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        # fully connected layer 3 ====================================
        params = {
            'a_out'         :    1000,
            'initializer'   :    self.initializer,
            'alpha'         :    self.alpha,
        }

        self.layers.append(FullyConnectedLayer(params=params))
        self.params.append(params)

        self.layers.append(ReLU())
        self.params.append(None)

        
        # fully connected layer 4 ====== my addition ====================================
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

        # backward prop
        for layer in reversed(self.layers):
            Z = layer.backward_prop(Z)['dx']

        pass


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