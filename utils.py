import cv2
import numpy as np
from keras.utils import np_utils
from keras.datasets import fashion_mnist as fm

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

def check_label():
    s = '     Label   Description\n\n' +\
        '0   T-shirt/top   T-shirt/上衣\n'+\
        '1   Trouser       褲子\n'+\
        '2   Pullover      毛衣\n'+\
        '3   Dress         連衣裙\n'+\
        '4   Coat          外套\n'+\
        '5   Sandal        涼鞋\n'+\
        '6   Shirt         襯衫\n'+\
        '7   Sneaker       運動鞋\n'+\
        '8   Bag           包包\n'+\
        '9   Ankle boot    靴子'
    print(s)
    
def img_transform(path):
    
    pic = cv2.imread(path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28, 28))

    pic = pic.reshape(-1, 28, 28, 1)
    pic = pic / 255.0
    
    return pic
        