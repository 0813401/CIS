import numpy as np
from keras.utils import np_utils
from keras.datasets import fashion_mnist as fm

def fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fm.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    classes = np.unique(y_train).shape[0]
#     print("classes:", classes)
    y_train = np_utils.to_categorical(y_train) 
    y_test = np_utils.to_categorical(y_test) 
    x_train, x_test = x_train / 255.0, x_test / 255.0
#     print("x_train shape:", x_train.shape)
#     print("y_train shape:", y_train.shape)

    return x_train, x_test, y_train, y_test, classes
