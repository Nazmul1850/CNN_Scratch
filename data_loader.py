import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def load_data(batch):
    img = [cv2.imread(path, 0) for path in './NumtaDB/' + batch['database name'] + '/' + batch['filename']]

    X_train = img
    Y_train = batch['digit']

    train_data = (X_train, Y_train)

    return train_data


def load_test_images(dir_path, filenames):
    img = [cv2.imread(dir_path + '/' + filename, 0) for filename in filenames]
    return img

