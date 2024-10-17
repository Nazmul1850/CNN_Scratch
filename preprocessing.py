import numpy as np
import cv2
from env import NUM_CLASSES

def preprocess_image(data):
    """
    Preprocess the image
    """
    if len(data) == 2:
        X, Y = data
    else:
        X = data

    X = [cv2.resize(img, (180, 180)) for img in X]

    # apply medianBlur
    X = [cv2.medianBlur(img, 3) for img in X]
    
    # apply threshold
    X = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in X]
    
    # apply dilation
    kernel = np.zeros((3, 3), np.uint8)
    X = [cv2.dilate(img, kernel, iterations=1) for img in X]

    # resize image
    X = [[cv2.resize(img, (28, 28))] for img in X]

    # invert image
    X = np.array(X)
    X = 255 - X
    X[X < 80] = 0.
    X[X >= 80] = 1. # normalize image


    if len(data) == 2:
        Y = one_hot_encode(np.array(Y))
        return (X, Y)
    else:
        return X
    

def one_hot_encode(Y, num_classes=NUM_CLASSES):
    """
    One hot encode the labels
    """
    Y = np.eye(num_classes)[Y.reshape(-1)]
    return Y




# import numpy as np
# import cv2
# NUM_CLASSES = 10

# def shapify_image(train_data, img_size=(180, 180)):
#     """
#     Reshape the image to img_size
#     """
#     X_train, Y_train = train_data
#     X_train = np.array([[cv2.resize(img, img_size)] for img in X_train])

#     return (np.array(X_train), np.array(Y_train)[:,np.newaxis])

# def preprocess_image(train_data):
#     """
#     Preprocess the image
#     """
#     X_train, Y_train = train_data
#     X_train = 255 - X_train
#     X_train[X_train < 100] = 0.
#     X_train[X_train >= 100] = 1.


#     Y_train = one_hot_encode(Y_train)

#     return (X_train, Y_train)

# def one_hot_encode(Y_train, num_classes=NUM_CLASSES):
#     """
#     One hot encode the labels
#     """
#     Y_train = np.eye(num_classes)[Y_train.reshape(-1)]
#     return Y_train
