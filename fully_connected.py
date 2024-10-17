import numpy as np
import math as m
from CNN_layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, params = {
        'a_out': 1,
        'initializer': 'he_uniform',
        'alpha': 0.001
    }):
        """
        Default Parameters
        ------------------
        params = {
            'a_out': 1,
            'initializer': 'he_uniform',
            'alpha': 0.001
        }

        Parameters
        ----------
        params : TYPE. python dictionary
            DESCRIPTION. contains the parameters required for fully connected layer
                a_out:     output dimensions
                initializer:    ['zero', 'random', 'xavier_unifrom', 'he_uniform', 'one']
                alpha:      learning rate

        Returns
        -------
        None.

        """
        self.a_out = params['a_out'] if 'a_out' in params else 1
        self.initializer = params['initializer'] if 'initializer' in params else 'he_uniform'
        self.alpha = params['alpha'] if 'alpha' in params else 0.001


        # params
        self.w = None
        self.b = None

        # input
        self.prev_layer = None
        
        pass


    def init_parameters(self):
        """
        initializes the parameters of the layer

        Returns
        -------
        None.

        """
        if self.prev_layer is None:
            print("Error: Previous layer is not set.")
            pass

        # previous layer shape
        _, a_in = self.prev_layer.shape

        if self.initializer == 'random':
            self.w = np.random.randn(self.a_out, a_in)
            self.b = np.random.randn(self.a_out, 1)
        elif self.initializer == 'xavier_uniform':
            self.w = np.random.uniform(-m.sqrt(6/(a_in)), m.sqrt(6/(a_in)), (self.a_out, a_in))
            self.b = np.random.uniform(-m.sqrt(6/(a_in)), m.sqrt(6/(a_in)), (self.a_out, 1))
        elif self.initializer == 'he_uniform':
            self.w = np.random.uniform(-m.sqrt(6/(a_in)), m.sqrt(6/(a_in)), (self.a_out, a_in))
            self.b = np.random.uniform(-m.sqrt(6/(a_in)), m.sqrt(6/(a_in)), (self.a_out, 1))
        elif self.initializer == 'one':
            self.w = np.ones((self.a_out, a_in))
            self.b = np.ones((self.a_out, 1))
        elif self.initializer == 'zero':
            self.w = np.zeros((self.a_out, a_in))
            self.b = np.zeros((self.a_out, 1))
        else :
            print("Error: Invalid initializer.")
            exit(0)

        pass
     


    def forward_prop(self, prev_layer):
        """
        forward propagation of fully connected layer

        Parameters
        ----------
        prev_layer : TYPE 2D numpy matrix
            DESCRIPTION. shape = (batchSize, input_dimension)

        Returns
        -------
        next_layer: 2D numpy matrix

        """

        prev_layer = np.array(prev_layer)
        self.prev_layer = prev_layer

        # initialize parameters
        if self.w is None:
            self.init_parameters()

        next_layer = prev_layer @ self.w.T + self.b.T
        next_layer = np.where(np.isnan(next_layer), 0., next_layer) # replace nan with 0
        
        return next_layer



    def backward_prop(self, dz):
        """
        backward propagation of fully connected layer

        Parameters
        ----------
        dz : TYPE  2D numpy matrix
            DESCRIPTION. next layer dz

        Returns
        -------
        dx for previous layer

        """

        dx = np.array(dz)
        dw = np.array(dz)
        db = np.array(dz)

        db = np.sum(dz, axis=0, keepdims=True).T/dz.shape[0]
        dw = dz.T @ self.prev_layer/dz.shape[0]
        dw = np.where(np.isnan(dw), 0., dw) # replace nan with 0

        dx = dz @ self.w
        dx = np.where(np.isnan(dx), 0., dx) # replace nan with 0

        assert dx.shape == self.prev_layer.shape
        assert dw.shape == self.w.shape
        assert db.shape == self.b.shape

        self.grads = {
            'dx': dx,
            'dw': dw,
            'db': db
        }
        
        # update parameters
        self.update_parameters()
        
        return self.grads       
        

    def update_parameters(self):
        """
        update parameters
        """
        self.w = self.w - self.alpha*self.grads['dw']
        self.b = self.b - self.alpha*self.grads['db']

        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        