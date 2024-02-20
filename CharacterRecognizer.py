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


    def __init__(self, model_path=None, encodings=None, input_shape=(64, 64, 3), num_classes=62, conv_architecture=[], dense_architecture=[], epochs=25, batch_size=64):
        self.model = self.load_model(model_path)
        self.encodings = encodings
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.conv_architecture = conv_architecture
        self.dense_architecure = dense_architecture
        self.epochs = epochs
        self.batch_size = batch_size
    

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
                resized_img = cv2.resize(img, self.input_shape[:2], interpolation=cv2.INTER_AREA)
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

    
    def build_model(self):
        # builds a CNN with the specified architecture
        model = Sequential()

        # add convolutional architecture
        for num_filters in self.conv_architecture:
            model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # flatten and add dense layers
        model.add(Flatten())
        for width in self.dense_architecture:
            model.add(Dense(width, activation='relu'))
        
        # add output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # compile the model
        model.compile(loss="categorical_crossentropy",
                    optimizer="Adam",
                    metrics=["accuracy"])
        
        # update the loaded model
        self.model = model
    

    def test_hyperparameters(self, num_conv_layers, min_conv_size, conv_increasing, num_dense_layers, min_dense_size, dense_increasing, num_epochs, batch_size):
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
        self.conv_architecture = conv_architecture
        self.dense_architecture = dense_architecture
        self.build_model()

        # train the model
        early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
        train_result = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=int(batch_size),
            epochs=int(num_epochs),
            verbose=1,
            callbacks=early_stopper
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
            f=self.test_hyperparameters,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        
        # maximize the validation accuracy
        optimizer.maximize(
            init_points=2,
            n_iter=1,
        )

        print(optimizer)
    

    def save_model(self, save_path="character_recognizer_model.h5"):
        # saves the model weights to the specified save path
        self.model.save_weights(save_path)
    

    def load_model(self, model_path):
        # loads the model weights from the specified model path
        if model_path is None:
            return None
        else:
            # build the model and load the weights
            self.build_model()
            self.model.load_weights(model_path)
        

if __name__ == "__main__":
    # initialize a CharacterRecognizer
    cr = CharacterRecognizer()

    # load data
    cr.load_dataset("character_dataset")
    print("Loaded Data!")

    # find optimal hyperparemeters
    cr.compute_optimal_hyperparamenters()