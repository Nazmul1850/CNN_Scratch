import numpy as np
from CNN_layer import Layer

class MaxPoolLayer(Layer):
    def __init__(self, params = {
        'filter_dim': (2, 2),
        'stride':   (1, 1)
    }):
        """
        Default Parameters
        ------------------
        params = {
            'filter_dim': (2, 2),
            'stride':   (1, 1)
        }

        Parameters
        ----------
        params : TYPE. python dictionary
            DESCRIPTION. contains the parameters required for max pooling
                filter_dim: Filter dimension (row, col)
                stride:     stride   (row, col)

        Returns
        -------
        None.

        """
        
        self.filter_dim = params['filter_dim'] if 'filter_dim' in params else (2,2)
        self.stride = params['stride'] if 'stride' in params else (1,1)
     
    def forward_prop(self, prev_layer):
        """
        applies max pooling on previous layer

        Parameters
        ----------
        prev_layer : TYPE 4D numpy matrix
            DESCRIPTION. shape = (batchSize, #ofChannels, #ofRows, #ofCols)

        Returns
        -------
        next_layer: 4D numpy matrix

        """
        
        
        prev_layer = np.array(prev_layer)
        batch, ch_in, row, col = prev_layer.shape

        # save previous layer for backprop
        self.prev_layer = prev_layer
        
        row_ = (row - self.filter_dim[0])//self.stride[0] + 1
        col_ = (col - self.filter_dim[1])//self.stride[1] + 1
        

        layer_strides = prev_layer.strides

        prev_layer_strided = np.lib.stride_tricks.as_strided(prev_layer, 
            shape=(batch, ch_in, row_, col_, self.filter_dim[0], self.filter_dim[1]),
            strides=(layer_strides[0], layer_strides[1], layer_strides[2]*self.stride[0], layer_strides[3]*self.stride[1], layer_strides[2], layer_strides[3]))


        next_layer = np.max(prev_layer_strided, axis=(4,5))
        
        return next_layer



    def backward_prop(self, dz):
        """
        backward propagation of maxpooling
        equally distributes gradient among all maximum value elements

        Parameters
        ----------
        dz : TYPE  4D numpy matrix
            DESCRIPTION. next layer dz

        Returns
        -------
        dx for previous layer

        """
        
        dz = np.array(dz)
        batch, ch_in, _, _ = self.prev_layer.shape
        _, _, row_, col_ = dz.shape
            
        dx = np.zeros(self.prev_layer.shape)

        for b in range(batch):
            for k in range(ch_in):
                for i in range(0, row_):
                    for j in range(0, col_):
                        arr_slice = self.prev_layer[b, k, i*self.stride[0]:i*self.stride[0]+self.filter_dim[0], j*self.stride[1]:j*self.stride[1]+self.filter_dim[1]]
                        max_val = np.max(arr_slice)

                        arr_slice[arr_slice != max_val] = 0.
                        arr_slice[arr_slice == max_val] = 1.
                        
                        dx[b, k, i*self.stride[0]:i*self.stride[0]+self.filter_dim[0], j*self.stride[1]:j*self.stride[1]+self.filter_dim[1]] = arr_slice*dz[b, k, i, j]/np.sum(arr_slice)
    

        assert dx.shape == self.prev_layer.shape, 'dx shape is not same as prev_layer shape'

        self.grads = {
            'dx': dx
        }
        
        return self.grads     
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        