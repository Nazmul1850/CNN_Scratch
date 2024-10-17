# import warnings
import numpy as np
from CNN_layer import Layer

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Softmax'
        self.activation = 'softmax'
        self.prev_layer = None
        
    def forward_prop(self, prev_layer):
        """
        Softmax activation function

        Arguments:
        prev_layer -- numpy array of shape (batch_size, n_classes)

        """

        self.prev_layer = prev_layer

        prev_layer = prev_layer - np.max(prev_layer, axis=1, keepdims=True)
        exps = np.exp(prev_layer)
        next_layer = exps / np.sum(exps, axis=1, keepdims=True)

        return next_layer
    

    def backward_prop(self, dz):
        """
        Backward propagation for softmax activation function

        Arguments:
        dz -- gradient of the cost with respect to the output of the softmax activation function
        
        """

        dx = np.array(dz, copy=True)

        self.grads = {
            'dx': dx,
        }

        return self.grads
