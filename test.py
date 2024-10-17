import numpy as np
import pandas as pd
from data_loader import load_data, load_test_images
from preprocessing import preprocess_image
from tqdm import tqdm
from alexnet import AlexNet
from lenet import LeNet
from testmodel import TestNet
import matplotlib.pyplot as plt

from convolution import ConvolutionLayer
from max_pooling import MaxPoolLayer
from flattening import FlatteningLayer
from fully_connected import FullyConnectedLayer
from softmax import Softmax
from relu_activation import ReLU
from evaluation_metrices import accuracy

import pickle
import sys

from env import *
import os

np.random.seed(SEED)

def main():

    path = sys.argv[1:][0]

    # create model
    # model = AlexNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    model = LeNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    # model = TestNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)

    # load pickle file
    with open(MODEL_FILENAME, 'rb') as f:
        trained_model = pickle.load(f)

    model.setLayers(trained_model)


#-------------------------------------------------------------------------------------------


    # Load data if tested with csv file
    
    # read csv file
    df_test = pd.read_csv(path)

    # remove unnecessary columns
    df_test.drop(['original filename', 'scanid', 'database name original', 'num'], axis=1, inplace=True)

    # split and reduce size
    df_test = df_test.sample(frac=TEST_SET_SECTION, random_state=SEED)
    filenames = df_test['filename'].values
    test_data = load_data(df_test)

    # print(df_test.shape)
#-------------------------------------------------------------------------------------------

# #-------------------------------------------------------------------------------------------

#     # load data if tested with directory path


#     # list directory files
#     filenames = os.listdir(path)
#     filenames = filenames[:int(len(filenames)*TEST_SET_SECTION)]

#     # load images
#     test_data = load_test_images(path, filenames)

#     # dummy test data y values, replace as required for accuracy calculation
#     test_data = (test_data, np.zeros(len(test_data), dtype=int))
# #-------------------------------------------------------------------------------------------

    # Preprocess data
    test_data = preprocess_image(test_data)

    print('test batch dimension', test_data[0].shape)


    # predict
    predictions = model.predict(test_data[0]).argmax(axis=1)
    print(predictions[:10])

    #write to csv
    df_pred = pd.DataFrame()
    df_pred['FileName'] = filenames
    df_pred['Digit'] = predictions
    df_pred.to_csv(PREDICTION_FILENAME, index=False)

    test_acc = model.evaluate(test_data)
    print('Test accuracy: {}'.format(test_acc))



if __name__ == '__main__':
    main()
