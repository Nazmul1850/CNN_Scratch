import numpy as np
import math as m
from CNN_layer import Layer


class ConvolutionLayer(Layer):
    def __init__(self, params):
        """
        

        Parameters
        ----------
        params : TYPE. python dictionary
            DESCRIPTION. contains the parameters required for convolution
                filter_dim: Filter dimension (row, col)
                stride:     stride   (row, col)
                padding:    padding   (row, col)
                ch_out:     Number of output channels
                initializer:    ['zero', 'random', 'xavier_unifrom', 'he_uniform', 'one']
                alpha:      learning rate
                
        Returns
        -------
        None.

        """
        
        self.filter_dim = params['filter_dim'] if 'filter_dim' in params else (3,3)
        self.stride = params['stride'] if 'stride' in params else (1,1)
        self.padding = params['padding'] if 'padding' in params else (0,0)
        self.ch_out = params['ch_out'] if 'ch_out' in params else 1
        self.initializer = params['initializer'] if 'initializer' in params else 'he_uniform'
        self.alpha = params['alpha'] if 'alpha' in params else 0.001

        # params
        self.w = None
        self.b = None

        # input
        self.prev_layer = None

        pass
        

    def init_parameters(self):

        if self.prev_layer is None:
            print("Error: Previous layer is not set.")
            exit(0)
        
        # previous layer shape 
        _, ch_in, _, _ = self.prev_layer.shape


        if self.initializer == 'random':
            self.w = np.random.randn(self.ch_out, ch_in, self.filter_dim[0], self.filter_dim[1])
            self.b = np.random.randn(self.ch_out, 1, 1)
        elif self.initializer == 'zero':
            self.w = np.zeros((self.ch_out, ch_in, self.filter_dim[0], self.filter_dim[1]))
            self.b = np.zeros((self.ch_out, 1, 1))
        elif self.initializer == 'one':
            self.w = np.ones((self.ch_out, ch_in, self.filter_dim[0], self.filter_dim[1]))
            self.b = np.ones((self.ch_out, 1, 1))
        elif self.initializer == 'xavier_uniform':
            self.w = np.random.uniform(-m.sqrt(6/(ch_in*self.filter_dim[0]*self.filter_dim[1])), m.sqrt(6/(ch_in*self.filter_dim[0]*self.filter_dim[1])), (self.ch_out, ch_in, self.filter_dim[0], self.filter_dim[1]))
            self.b = np.random.uniform(-m.sqrt(6/(ch_in*self.filter_dim[0]*self.filter_dim[1])), m.sqrt(6/(ch_in*self.filter_dim[0]*self.filter_dim[1])), (self.ch_out, 1, 1))
        elif self.initializer == 'he_uniform':
            self.w = np.random.uniform(-m.sqrt(6/(ch_in)), m.sqrt(6/(ch_in)), (self.ch_out, ch_in, self.filter_dim[0], self.filter_dim[1]))
            self.b = np.random.uniform(-m.sqrt(6/(ch_in)), m.sqrt(6/(ch_in)), (self.ch_out, 1, 1))
        else:
            print("Error: Invalid initializer")
            exit(0)

        pass


    def forward_prop(self, prev_layer):
        """
        applies convolution on previous layer

        Parameters
        ----------
        prev_layer : TYPE 4D numpy matrix
            DESCRIPTION. shape = (batchSize, #ofChannels, #ofRows, #ofCols)

        Returns
        -------
        next_layer: 3D numpy matrix

        """
        
        # pad with zeros
        prev_layer = np.pad(np.array(prev_layer), ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))

        # pad with edge values
        # prev_layer = np.pad(np.array(prev_layer), ((0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), 'edge')

        # previous layer shape 
        batch, ch_in, row, col = prev_layer.shape

        # save layer input
        self.prev_layer = prev_layer

        # initialize parameters if not initialized
        if self.w is None:
            self.init_parameters()

        # next layer
        row_ = (row - self.filter_dim[0])//self.stride[0] + 1
        col_ = (col - self.filter_dim[1])//self.stride[1] + 1
        

        layer_strides = prev_layer.strides
        prev_layer_strided = np.lib.stride_tricks.as_strided(prev_layer, 
            shape=(batch, ch_in, row_, col_, self.filter_dim[0], self.filter_dim[1]),
            strides=(layer_strides[0], layer_strides[1], layer_strides[2]*self.stride[0], layer_strides[3]*self.stride[1], layer_strides[2], layer_strides[3]))


        dot = np.tensordot(prev_layer_strided, self.w, axes=((1, 4, 5), (1, 2, 3)))
        next_layer = np.transpose(dot, (0, 3, 1, 2)) + self.b

        return next_layer
            
        
    def backward_prop(self, dz):
        """
        backward propagation of convolution

        Parameters
        ----------
        dz : TYPE  4D numpy matrix
            DESCRIPTION. next layer dz

        Returns
        -------
        dx for previous layer

        """
        
        dz = np.array(dz)
        batch, _, row, col = dz.shape
        
            
        dx = np.zeros(self.prev_layer.shape)
        dw = np.zeros(self.w.shape)
        
        # calculating db
        db = np.sum(dz, axis=(0, 2, 3)).reshape(self.b.shape)

        # calculating dw
        dz_sparsed_row = self.stride[0]*(row -1) + 1
        dz_sparsed_col = self.stride[1]*(col -1) + 1
        dz_sparsed = np.zeros(( batch, self.ch_out, dz_sparsed_row, dz_sparsed_col))
        dz_sparsed[:, :, ::self.stride[0], ::self.stride[1]] = dz

        _, ch_in, row_, col_ = dw.shape
        
        layer_strides = self.prev_layer.strides
        prev_layer_strided = np.lib.stride_tricks.as_strided(self.prev_layer, 
            shape=(batch, ch_in, row_, col_, dz_sparsed_row, dz_sparsed_col),
            strides=(layer_strides[0], layer_strides[1], layer_strides[2], layer_strides[3], layer_strides[2], layer_strides[3]))


        dot = np.tensordot(prev_layer_strided, dz_sparsed, axes=((0, 4, 5), (0, 2, 3)))
        dw = np.transpose(dot, (3, 0, 1, 2))




        # calculating dx
        pad_0 = int(self.filter_dim[0]-1)
        pad_1 = int(self.filter_dim[1]-1)
        dz_sparsed_paded = np.pad(dz_sparsed, ((0, 0), (0, 0), (pad_0, pad_0) , (pad_1, pad_1)))
        w_rotated = np.flip(np.flip(self.w, 2), 3)
        

        _, _, row_, col_ = dz_sparsed_paded.shape
        _, ch_in, w_rotated_row, w_rotated_col = w_rotated.shape


        dz_sp_pad_strides = dz_sparsed_paded.strides
        dz_sparsed_paded_strided = np.lib.stride_tricks.as_strided(dz_sparsed_paded, 
            shape=(batch, self.ch_out, row_ - w_rotated_row + 1, col_ - w_rotated_col + 1, w_rotated_row, w_rotated_col),
            strides=(dz_sp_pad_strides[0], dz_sp_pad_strides[1], dz_sp_pad_strides[2], dz_sp_pad_strides[3], dz_sp_pad_strides[2], dz_sp_pad_strides[3]))


        dot = np.tensordot(dz_sparsed_paded_strided, w_rotated, axes=((1, 4, 5), (0, 2, 3)))
        dot = np.transpose(dot, (0, 3, 1, 2))

        dx[:, :, :dot.shape[2], :dot.shape[3]] = dot   # adjust dx shape
        dx = dx[:, :, self.padding[0]:dx.shape[2]-self.padding[0], self.padding[1]:dx.shape[3]-self.padding[1]]   # unpad dx


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

        Returns
        -------
        None.

        """
        self.w = self.w - self.alpha*self.grads['dw']
        self.b = self.b - self.alpha*self.grads['db']

        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        