import os
import csv
import cv2

# model building imports
import keras                              
from keras.models import Sequential     
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def load_dataset(folder_path):
    print("Loading Data...")
    X_train, y_train = [], []
    label_counter = {}
    with open(os.path.join(folder_path, "labels.csv")) as f:
        labels_reader = csv.reader(f)
        header = next(labels_reader)
        for row in labels_reader:
            img_path, label = row
            if label not in label_counter:
                label_counter[label] = 0
            label_counter[label] += 1
            if label_counter[label] > 5:
                continue
            X_train.append(cv2.imread(os.path.join(folder_path, img_path)))
            y_train.append(label)
    return X_train, y_train


def build_cnn(input_shape, conv_architecture, dense_architecture):
    # builds a Convolutional Neural Network using keras
    print("Building CNN...")
    model = Sequential()

    # add convolutional layers
    for num_filters in conv_architecture:
        model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.05))
    
    # flatten and add dense layers
    model.add(Flatten())
    for width in dense_architecture:
        model.add(Dense(width, activation='relu'))
        model.add(Dropout(0.05))
    
    # add output layer
    model.add(Dense(3, activation='softmax'))

    # compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adagrad",
                  metrics=["accuracy"])
    
    # return the built and compiled model
    return model


if __name__ == "__main__":
    # load data
    X_train, y_train = load_dataset("character_dataset")
    print(len(y_train))
    exit()

    # set hyperparameters
    num_classes = 62
    input_shape = X_train[0].shape
    conv_architecture = [128, 256]
    dense_architecture = [512]
    num_epochs = 30
    batch_size = 128

    # build the CNN with optimal hyperparameters
    cnn = build_cnn(input_shape, conv_architecture, dense_architecture)

    # train the model
    # ensure the training stops if there is no improvement for 2 epochs
    early_stopper = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    train_result = cnn.fit(X_train,
                           y_train,
                           validation_split=0.1,
                           batch_size=batch_size,
                           epochs=num_epochs,
                           verbose=1,
                           callbacks=[early_stopper])


