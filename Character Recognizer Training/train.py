import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
# model building imports
import keras                              
from keras.models import Sequential     
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def load_dataset(folder_path):
    # loads dataset
    print("Loading Data...")
    X_train, y_train = [], []
    one_hot_encoding = {}
    with open(os.path.join(folder_path, "labels.csv")) as f:
        labels_reader = csv.reader(f)
        header = next(labels_reader)
        for row in labels_reader:
            img_path, label = row
            img = cv2.imread(os.path.join(folder_path, img_path))
            resized_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            X_train.append(resized_img)
            if label not in one_hot_encoding:
                one_hot_encoding[label] = len(one_hot_encoding)
            new_label = [0][:]*62
            new_label[one_hot_encoding[label]] = 1
            y_train.append(new_label)
    X_train, y_train = np.array(X_train).astype('float64'), np.array(y_train).astype('int')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    return X_train, X_val, y_train, y_val, one_hot_encoding


def build_cnn(input_shape, conv_architecture, dense_architecture, num_classes):
    # builds a Convolutional Neural Network using keras
    model = Sequential()

    # add convolutional architecture
    for num_filters in conv_architecture:
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # flatten and add dense layers
    model.add(Flatten())
    for width in dense_architecture:
        model.add(Dense(width, activation='relu'))
    
    # add output layer
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])
    
    # return the built and compiled model
    return model


def test_model(num_conv_layers, min_conv_size, conv_increasing, num_dense_layers, min_dense_size, dense_increasing, num_epochs, batch_size):
    # tests a model with the given hyperparameters
    # set constants
    num_classes = 62
    input_shape = X_train[0].shape
    #print('a')
    #print(num_conv_layers, min_conv_size, conv_increasing, num_dense_layers, min_dense_size, dense_increasing, num_epochs, batch_size)
    #print('a')

    # create convolutional architecture
    conv_architecture = [int(min_conv_size * (2**i)) for i in range(int(num_conv_layers))]
    if not conv_increasing:
        conv_architecture.reverse()
    
    # create dense architecture
    dense_architecture = [int(min_dense_size * (2**i)) for i in range(int(num_dense_layers))]
    if not dense_increasing:
        dense_architecture.reverse()
    #6         | 0.7419    | 38.15     | 0.0       | 0.0       | 70.23     | 167.1     | 4.0       | 1.0       | 50.0      |
    # build the model
    cnn = build_cnn(input_shape, conv_architecture, dense_architecture, num_classes)

    # train the model
    train_result = cnn.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=int(batch_size),
        epochs=int(num_epochs),
        verbose=20
    )

    # return the validation accuracy
    return train_result.history['val_accuracy'][-1]



if __name__ == "__main__":
    # load data
    X_train, X_val, y_train, y_val, one_hot_encoding = load_dataset("character_dataset")
    print("Loaded", len(y_train) + len(y_val), "images!")

    # find optimal hyperparameters
    pbounds = {
        'num_conv_layers': (1, 4), 
        'min_conv_size': (4, 128),
        'conv_increasing': (0, 1),
        'num_dense_layers': (1, 4), 
        'min_dense_size': (4, 512),
        'dense_increasing': (0, 1),
        'num_epochs': (10, 50),
        'batch_size': (1, 128)
    }
    optimizer = BayesianOptimization(
        f=test_model,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(
        init_points=2,
        n_iter=30,
    )


