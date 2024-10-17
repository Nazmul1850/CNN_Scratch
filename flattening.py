import numpy as np
from CNN_layer import Layer

class FlatteningLayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward_prop(self, prev_layer):
        """
        Flattens the input matrix

        Parameters
        ----------
        prev_layer : TYPE 4D numpy matrix
            DESCRIPTION. shape = (batchSize, #ofChannels, #ofRows, #ofCols)

        Returns
        -------
        next_layer: 2D numpy matrix

        """
        prev_layer = np.array(prev_layer)
        batch, ch_in, row, col = prev_layer.shape

        # save previous layer for backprop
        self.prev_layer = prev_layer

        next_layer = prev_layer.reshape(batch, ch_in*row*col)

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
        batch, ch_in, row, col = self.prev_layer.shape

        dx = dx.reshape(batch, ch_in, row, col)

        self.grads = {
            'dx': dx
        }

        return self.grads