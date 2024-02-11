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


class CharacterRecognizer(object):


    def __init__(self, model_path=None, encodings=None, input_shape=(64, 64), num_classes=62):
        self.model = self.load_model_path(model_path)
        self.encodings = self.encodings
        self.input_shape = self.input_shape
        self.num_classes = self.num_classes
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    

    def load_dataset(self, dataset_path, test_size=0.1):
        # initialize train data and encoding map
        X_train, y_train = [], []
        one_hot_encoding = {}

        # read the labels csv inside the dataset path
        with open(os.path.join(dataset_path, "labels.csv")) as f:
            # read the header
            labels_reader = csv.reader(f)
            header = next(labels_reader)
            for row in labels_reader:
                # read the class label and load the image with cv2
                img_path, label = row
                img = cv2.imread(os.path.join(dataset_path, img_path))
                resized_img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_AREA)
                X_train.append(resized_img)

                # convert the class label to one hot encoding
                if label not in one_hot_encoding:
                    one_hot_encoding[label] = len(one_hot_encoding)
                new_label = [0][:]*self.num_classes
                new_label[one_hot_encoding[label]] = 1
                y_train.append(new_label)
        
        # initialize train data as np.array's and split into train and val
        X_train, y_train = np.array(X_train).astype('float64'), np.array(y_train).astype('int')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size)

        # update train data and encoding
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.encodings = one_hot_encoding

    
    def build_model(self, conv_architecture, dense_architecture):
        # builds a CNN with the specified architecture
        model = Sequential()

        # add convolutional architecture
        for num_filters in conv_architecture:
            model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # flatten and add dense layers
        model.add(Flatten())
        for width in dense_architecture:
            model.add(Dense(width, activation='relu'))
        
        # add output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # compile the model
        model.compile(loss="categorical_crossentropy",
                    optimizer="Adam",
                    metrics=["accuracy"])
        
        # update the loaded model
        self.model = model
    

    def test_hyperparameters(num_conv_layers, min_conv_size, conv_increasing, num_dense_layers, min_dense_size, dense_increasing, num_epochs, batch_size):
        # computes the validation accuracy of the model trained with the specified hyperparameters
        num_classes = self.num_classes
        input_shape = self.X_train[0].shape
    
        # create convolutional architecture
        conv_architecture = [int(min_conv_size * (2**i)) for i in range(int(num_conv_layers))]
        if not conv_increasing:
            conv_architecture.reverse()
        
        # create dense architecture
        dense_architecture = [int(min_dense_size * (2**i)) for i in range(int(num_dense_layers))]
        if not dense_increasing:
            dense_architecture.reverse()

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


    def compute_optimal_hyperparamenters(self):
        # computes the hyperparameters that maximize validation accuracy using Bayesian Optimization
        # set bounds for hyperparameters (based on past experience with character recognizers)
        pbounds = {
            'num_conv_layers': (1, 3), 
            'min_conv_size': (32, 128),
            'conv_increasing': (0, 1),
            'num_dense_layers': (1, 3), 
            'min_dense_size': (32, 512),
            'dense_increasing': (0, 1),
            'num_epochs': (25, 25),
            'batch_size': (64, 64)
        }

        # initialize the optimizer
        optimizer = BayesianOptimization(
            f=test_model,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        
        # maximize the validation accuracy
        optimizer.maximize(
            init_points=2,
            n_iter=30,
        )






if __name__ == "__main__":
    # initialize a CharacterRecognizer
    cr = CharacterRecognizer()

    # load data
    cr.load_dataset("character_dataset")

    # find optimal hyperparemeters
    cr.compute_optimal_hyperparamenters()