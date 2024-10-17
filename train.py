import numpy as np
import pandas as pd
from data_loader import load_data
from preprocessing import preprocess_image
from tqdm import tqdm
from alexnet import AlexNet
from lenet import LeNet
from testmodel import TestNet
import matplotlib.pyplot as plt
from evaluation_metrices import *
import sklearn.metrics as metrics

from env import *

np.random.seed(SEED)

def main():

    # create model
    # model = AlexNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    model = LeNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    # model = TestNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)

    # Load data
    
    # use pandas to read csv file
    df_a = pd.read_csv('./NumtaDB/training-a.csv')
    df_b = pd.read_csv('./NumtaDB/training-b.csv')
    df_c = pd.read_csv('./NumtaDB/training-c.csv')

    # remove unnecessary columns
    df_a.drop(['original filename', 'scanid', 'database name original', 'contributing team'], axis=1, inplace=True)
    df_b.drop(['original filename', 'scanid', 'database name original', 'contributing team'], axis=1, inplace=True)
    df_c.drop(['original filename', 'scanid', 'database name original', 'contributing team'], axis=1, inplace=True)

    # print(df_a.shape)
    # print(df_b.shape)
    # print(df_c.shape)

    # df joined
    df = pd.concat([df_b, df_a, df_c], ignore_index=True)

    # df shuffled
    df = df.sample(frac=1).reset_index(drop=True)
    
    # df split
    df_train = df.iloc[:int(df.shape[0]*0.8), :]
    df_val = df.iloc[int(df.shape[0]*0.8):, :]

    
    # df_train = df.iloc[:int(df.shape[0]*TRAIN_SET_SECTION), :]
    # df_val = df.iloc[int(df.shape[0]*TRAIN_SET_SECTION):int(df.shape[0]*TRAIN_SET_SECTION)+int(df.shape[0]*VAL_SET_SECTION), :]
    
    print('Train set:', df_train.shape[0])
    print('Validation set:', df_val.shape[0])


    batches = np.array_split(df_train, df_train.shape[0]//BATCH_SIZE)
    batches.append(df_train.iloc[BATCH_SIZE*(df_train.shape[0]//BATCH_SIZE):, :])


    # prepare validation data
    # load images
    val_data = load_data(df_val) 

    # Preprocess data
    val_data = preprocess_image(val_data)

    train_crossEntropyLoss = []

    val_crossEntropyLoss = []
    val_macro_f1 = []
    val_accuracy = []

    total_batches = len(batches)
    loss = 0

    for epoch in tqdm(range(TOTAL_EPOCHS), desc="Training", leave=False):
        for i in tqdm(range(total_batches), desc='Epoch {}/{}'.format(epoch+1, TOTAL_EPOCHS), leave=False):
            # load images
            train_data = load_data(batches[i]) # (X_train, y_train) not numpy

            # Preprocess data
            train_data = preprocess_image(train_data)

            loss = model.train(train_data)
            model.save()

        
        print()
        print('Loss: {}'.format(loss))
        train_crossEntropyLoss.append(loss)

        # validation
        val_pred = model.predict(val_data[0])

        y_true = val_data[1].argmax(axis=1)
        y_pred = val_pred.argmax(axis=1)

        val_acc = metrics.accuracy_score(y_true, y_pred)
        val_loss = model.calc_loss(val_pred, val_data[1])
        val_f1 = metrics.f1_score(y_true, y_pred, average='macro', labels=range(NUM_CLASSES))
        
        print('Validation accuracy: {}'.format(val_acc))
        print('Validation loss: {}'.format(val_loss))
        print('Validation macro f1 score: {}'.format(val_f1))
        print()

        val_crossEntropyLoss.append(val_loss)
        val_accuracy.append(val_acc)
        val_macro_f1.append(val_f1)


    print('Training done ------------------')


    # plot loss vs epoch
    plt.figure(1)
    plt.plot(val_crossEntropyLoss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_epoch.png')
    plt.show()
    plt.close()

    
    # plot accuracy vs epoch
    plt.figure(2)
    plt.plot(val_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_vs_epoch.png')    
    plt.show()
    plt.close()
    
    # plot f1 vs epoch
    plt.figure(3)
    plt.plot(val_macro_f1)
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1 score')
    plt.savefig('f1_vs_epoch.png')
    plt.show()
    plt.close()

    # predict
    print(model.predict(np.array([train_data[0][0]])), train_data[1][0])

    #   print model summary

    print('accuracy:', val_accuracy)
    print('loss:', val_crossEntropyLoss)
    print('f1:', val_macro_f1)


    # save model
    model.save()

    # might get error
    print('Confusion matrix:')
    print(metrics.confusion_matrix(y_true, y_pred))



if __name__ == '__main__':
    main()
