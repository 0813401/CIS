import numpy as np
from keras.utils import np_utils
from keras.datasets import fashion_mnist as fm
from sklearn.model_selection import train_test_split

# # # # # # # # # # # # #
#      fashion mnist    #
# # # # # # # # # # # # #

def fashion_mnist(split = True):
    (x_train, y_train), (x_test, y_test) = fm.load_data()
    if split:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        classes = np.unique(y_train).shape[0]
        y_train = np_utils.to_categorical(y_train) 
        y_test = np_utils.to_categorical(y_test) 
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, x_test, y_train, y_test, classes
    else:
        x = np.concatenate((x_train, x_test), axis = 0)
        y = np.concatenate((y_train, y_test), axis = 0)
        classes = np.unique(y).shape[0]
        y = np_utils.to_categorical(y) 
        x = x / 255.0
        
        return x, y, classes
    
# # # # # # # # # # # # #
#   bigbasket_products  #
# # # # # # # # # # # # #

# def bigbasket_products():
#     npz_file = np.load('bigbasket_products.npz')
#     x_train, x_test, y_train, y_test = npz_file['x1'], npz_file['x2'], npz_file['y1'], npz_file['y2']
#     classes = len(y_train[0])
    
#     return x_train, x_test, y_train, y_test, classes 

def bigbasket_products():
    x = np.load('data.npy', allow_pickle = True) # 這裡你可能要自己改
    y = np.load('data_label.npy', allow_pickle = True) # 這裡你可能要自己改
    classes = len(y[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return x_train, x_test, y_train, y_test, classes 
