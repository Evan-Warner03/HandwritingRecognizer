import os
import csv
import cv2
import json
import numpy as np
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# model building imports
import keras                              
from keras.models import Sequential     
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# surpress warnings
import warnings
warnings.filterwarnings('ignore')


class CharacterRecognizer(object):


    def __init__(self, model_path=None, encodings=None, input_shape=(32, 32, 1), num_classes=62, conv_architecture=[], dense_architecture=[], epochs=25, batch_size=64):
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
    

    def preprocess_image(self, image):
        # resizes image to expected size and normalizes pixel value to [0, 1]
        # first find and crop image to largest shape only
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, grayscale = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(grayscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Find object with the biggest bounding box
        mx = (0,0,0,0)
        mx_area = 0
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            area = w*h
            if area > mx_area:
                mx = x,y,w,h
                mx_area = area
        x,y,w,h = mx
        cropped = image[y:y+h, x:x+w]

        # resize the image
        resized = cv2.resize(cropped, tuple(self.input_shape[:2]), interpolation=cv2.INTER_AREA)

        # normalize the image
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, grayscale = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
        normalized = grayscale / 255.0
        return normalized
    

    def load_dataset(self, dataset_path, test_size=0.1):
        # initialize train data and encoding map
        X_train, y_train = [], []
        one_hot_encoding = {}

        # read the labels csv inside the dataset path
        with open(os.path.join(dataset_path, "labels.csv")) as f:
            # read the header
            labels_reader = csv.reader(f)
            header = next(labels_reader)
            for row in tqdm(labels_reader):
                # read the class label and load the image with cv2
                img_path, label = row
                img = cv2.imread(os.path.join(dataset_path, img_path))
                processed_img = self.preprocess_image(img)
                X_train.append(processed_img)

                # convert the class label to one hot encoding
                if label not in one_hot_encoding:
                    one_hot_encoding[label] = len(one_hot_encoding)
                new_label = [0][:]*self.num_classes
                new_label[one_hot_encoding[label]] = 1
                y_train.append(new_label)
        
        # initialize train data as np.array's and split into train and val
        X_train, y_train = np.array(X_train).astype('float64'), np.array(y_train).astype('int')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size)

        # transpose the encodings so we can convert from prediction to character in the future
        transposed_encodings = {}
        for encoding in one_hot_encoding:
            transposed_encodings[one_hot_encoding[encoding]] = encoding

        # update train data and encoding
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.encodings = transposed_encodings

    
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
        early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
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


    def compute_optimal_hyperparameters(self):
        # computes the hyperparameters that maximize validation accuracy using Bayesian Optimization
        # set bounds for hyperparameters (based on past experience with character recognizers)
        cr.test_hyperparameters(
            num_conv_layers=3, 
            min_conv_size=128, 
            conv_increasing=1, 
            num_dense_layers=1, 
            min_dense_size=512, 
            dense_increasing=1, 
            num_epochs=25, 
            batch_size=64
        )
        pbounds = {
            'num_conv_layers': (3, 5), 
            'min_conv_size': (128, 182),
            'conv_increasing': (1, 1),
            'num_dense_layers': (1, 2), 
            'min_dense_size': (512, 1024),
            'dense_increasing': (1, 1),
            'num_epochs': (25, 25),
            'batch_size': (32, 128)
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
            n_iter=10,
        )

        print(optimizer.max)
    

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
    

    def save_hyperparameters(self, save_path="character_recognizer_hp.json"):
        # saves the model hyperparameters to a json
        hyperparameters = {
            "conv_architecture": self.conv_architecture,
            "dense_architecture": self.dense_architecture,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        with open(save_path, "w") as f:
            json.dump(hyperparameters, f, indent=4)
    

    def load_hyperparameters(self, hyperparameter_path):
        # loads the model hyperparameters from a json
        with open(hyperparameter_path) as f:
            hyperparameters = json.load(f)
        self.conv_architecture = hyperparameters["conv_architecture"]
        self.dense_architecture = hyperparameters["dense_architecture"]
        self.num_classes = hyperparameters["num_classes"]
        self.input_shape = hyperparameters["input_shape"]
        self.epochs = hyperparameters["epochs"]
        self.batch_size = hyperparameters["batch_size"]

    
    def save_encodings(self):
        # saves encoding to json
        with open("character_recognizer_encodings.json", "w") as f:
            json.dump(self.encodings, f, indent=4)
    

    def load_encodings(self, encodings_path):
        # loads encodings from json
        with open(encodings_path) as f:
            self.encodings = json.load(f)
    

    def classify_characters(self, characters, preprocess=True):
        # returns the list of classified characters
        # first resize all of the images
        if preprocess:
            for i in range(len(characters)):
                characters[i] = self.preprocess_image(characters[i])

        # generate the predictions
        characters = np.array(characters)
        preds = self.model.predict(characters)

        # convert the predictions back to text via encodings
        final_predictions = []
        for p in preds:
            final_predictions.append([self.encodings[str(np.argmax(p))], p[np.argmax(p)]])

        return final_predictions
        

if __name__ == "__main__":
    # initialize a CharacterRecognizer
    cr = CharacterRecognizer()
    optimize = False

    # load data
    print("Loading Data...")
    cr.load_dataset("character_dataset")
    cr.save_encodings()
    print("Loaded Data!")

    if optimize:
        # find optimal hyperparemeters
        cr.compute_optimal_hyperparameters()
    else:
        # build model with optimal hyperparameters
        cr.test_hyperparameters(
            num_conv_layers=5,
            min_conv_size=128,
            conv_increasing=1,
            num_dense_layers=1,
            min_dense_size=512,
            dense_increasing=1,
            num_epochs=25,
            batch_size=64
        )

        # save model and hyperparameters
        cr.save_model()
        cr.save_hyperparameters()