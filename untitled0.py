from max_pooling import MaxPoolLayer
from convolution import ConvolutionLayer
from fully_connected import FullyConnectedLayer
from flattening import FlatteningLayer
from relu_activation import ReLU
from softmax import Softmax

import matplotlib.pyplot as plt


import numpy as np

m = [1, 1, 1]

m = np.array(m)

# m = np.array([m, 2*m, 3*m, 4*m])
# m = np.array([m, 2*m, 3*m, 4*m])
# m[0, 1, 1] = 2


# print(m)
# print(np.average(m))
# print(m.shape)


# ch, row, col = m.shape

# print(np.zeros((2, 3, 4, 5)))

# A = np.array([m, m])
# A = np.array([A, A])
# B = np.concatenate((A, A*2), axis=2)
# B = np.concatenate((B, B), axis=1)
# B = np.array([B, B])

# # print(np.expand_dims(m,axis=0))
# print(A)
# print(B)

# random integer
B = np.random.randint(-10, 10, size=(2, 2, 4, 5))

# print(B)

batch, channel, row, col = B.shape

stride = 2
kernel_size = 2

B_strides = B.strides

# # as_strided stride more than 1
# D = np.lib.stride_tricks.as_strided(B, shape=(batch, channel, (row - kernel_size)//stride + 1, (col - kernel_size)//stride + 1, kernel_size, kernel_size),
#         strides=(B_strides[0], B_strides[1], B_strides[2]*stride, B_strides[3]*stride, B_strides[2], B_strides[3]))

# print(D)


# print()
# print(np.max(D, axis=(4, 5)))



print()
print()

# sparse B matrix by stride 
# B_sparse = np.zeros(( batch, channel, 2*row -1, 2*col -1))
# B_sparse[:, :, ::2,::2] = B
# print(B_sparse)



# kernel = np.ones((channel, 2, 2))
# kernels = np.array([kernel, kernel*2])

# bias = np.ones((channel, 1, 1))
# print(bias)
# # print(kernels)


# print()
# print()
# dot1 = np.tensordot(D, kernel, axes=((1, 4, 5), (0, 1, 2)))
# dot2 = np.tensordot(D, kernels, axes=((1, 4, 5), (1, 2, 3)))
# print(dot1)
# # print(dot2)

# print('D shape', D.shape)
# print('kernels shape', kernels.shape)
# print('dot1 shape', dot1.shape)
# print('dot2 shape', dot2.shape)

# # change axis of dot2
# dot3 = np.transpose(dot2, (0, 3, 1, 2)) + bias
# print(dot3)


# padding test

# print(np.sum(B, axis=(1, 2)))

# test = np.array(
#     [
#         [
#             [1, 2],
#             [3, 4]
#         ],
#         [
#             [5, 6],
#             [7, 8]
#         ]
#     ]
# )
# print('here', np.flip(np.flip(test, 1), 2))


# print(np.pad(B, ((0, 0), (2, 1), (1, 3))))

# print(A.shape)

# C = np.tensordot(A, B[:, :, 3:], axes=((0, 1, 2),(0, 1, 2)))
# print('here', C)
# print(C.shape)

# print(np.diag(C))


# ==============================================================================

# convolution test

# params = {
#     'filter_dim': (2, 2),
#     'stride'    : (2, 2),
#     'padding'   : (0, 0),
#     'ch_out'    : 4,
#     'initializer': 'xavier',
# }
# conv = ConvolutionLayer(params=params)

# front = conv.forward_prop(B)

# # print(front)
# back = conv.backward_prop(np.ones(front.shape))


# print(back['dx'].shape)
# print(B.shape)

# ==============================================================================




# =============================================================================
# 
# max pooling test

# stride = (2, 2)
# pool_size = (2, 2) # = (row, col)

# params = {
#     'filter_dim': (2, 2),
#     'stride':   stride,
#     }

# maxpool = MaxPoolLayer(params)
# next_layer = maxpool.forward_prop(B)


# dz = np.ones(next_layer.shape)

# # print(dz)
# dx = maxpool.backward_prop(dz)


# print('next_layer:', next_layer)
# print('dx:', dx)
# =============================================================================



# =============================================================================

# fully connected test

# params = {
#     'a_out': 10,
#     'initializer': 'xavier',
#     'alpha': 0.1,
# }

# fc = FullyConnectedLayer(params)

# X = np.random.randint(0, 10, size=(10, 5))

# Z = fc.forward_prop(X)

# dz = np.ones(Z.shape)

# grad = fc.backward_prop(dz)

# print('Z:', Z)
# print('grad:', grad)

# fc.update_parameters()



# =============================================================================


# =============================================================================
# # flatten test

# flatLayer = FlatteningLayer()

# Z = flatLayer.forward_prop(B)
# X = flatLayer.backward_prop(Z)['dx']

# print('Z:', Z)
# print('X:', X)
# print('B:', B)

# assert np.array_equal(X, B)

# =============================================================================



# =============================================================================
# # relu test

# relu = ReLU()

# Z = relu.forward_prop(B)
# X = relu.backward_prop(Z)['dx']

# print('Z:', Z)
# print('X:', X)
# print('B:', B)

# =============================================================================



# =============================================================================
# # soft test

# soft = Softmax()

# Z = soft.forward_prop(B)
# X = soft.backward_prop(Z)['dx']

# print('Z:', Z)
# print('X:', X)
# print('B:', B)

# =============================================================================








# X = np.random.randint(-10, 10, size=(10,))
# Y = np.random.randint(-10, 10, size=(10,))


# # plot f1 vs epoch
# plt.figure(3)
# plt.plot(X, 'r', label='Macro F1 score')
# plt.xlabel('Epochs')
# plt.ylabel('Macro F1 score')
# plt.savefig('f1_vs_epoch.png')
# plt.show()
# plt.close()

# plt.figure(4)
# plt.plot(Y, 'b', label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig('loss_vs_epoch.png')
# plt.show()
# plt.close()

from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 3]
print(f1_score(y_true, y_pred, average='macro'))

# with labels argument
print(f1_score(y_true, y_pred, labels=range(3), average='macro'))

# confusion matrix
from sklearn.metrics import confusion_matrix
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

# def one_hot(y):
#     y_one_hot = np.zeros((len(y), 3))
#     for i in range(len(y)):
#         y_one_hot[i, y[i]] = 1
#     return y_one_hot

# y_true = one_hot(y_true)
# y_pred = one_hot(y_pred)

print(confusion_matrix(y_true, y_pred))
