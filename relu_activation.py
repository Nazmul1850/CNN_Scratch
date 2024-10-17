import numpy as np
from CNN_layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'
        self.activation = 'relu'
        self.prev_layer = None
        
    def forward_prop(self, prev_layer):
        self.prev_layer = prev_layer
        return np.maximum(0, prev_layer)
    
    def backward_prop(self, dz):
        dx = np.array(dz, copy=True)
        dx[self.prev_layer <= 0] = 0

        self.grads = {
            'dx': dx
        }
        
        return self.grads