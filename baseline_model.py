# # # # # # # # # #
#      module     #
# # # # # # # # # #
import keras
import keras.backend as K
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import add, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Input, ZeroPadding2D, BatchNormalization, AveragePooling2D


# # # # # # # # # #
#     resnet50    #
# # # # # # # # # #
def resnet50(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False):

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last': 
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' +  str(stage) +  block +  '_branch'
        bn_name_base = 'bn' +  str(stage) +  block +  '_branch'
        x = Conv2D(filters1, (1, 1), name = conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis = bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
        x = add([x, input_tensor])
        x = Activation('relu')(x)

        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' +  str(stage) +  block +  '_branch'
        bn_name_base = 'bn' +  str(stage) +  block +  '_branch'
        x = Conv2D(filters1, (1, 1), strides=strides, name = conv_name_base  + '2a')(input_tensor)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base  + '2a')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base  + '2b')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base  + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), name = conv_name_base  + '2c')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base  + '2c')(x)
        shortcut = Conv2D(filters3, (1, 1), strides=strides, name = conv_name_base  + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base  + '1')(shortcut)
        x = add([x, shortcut])
        x = Activation('relu')(x)

        return x
    
    img_input = Input(shape = (height, width, depth))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    x = ZeroPadding2D((3, 3))(img_input) 
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x) 
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x) 
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #stage2#
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    #stage3#
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #stage4#
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    #stage5#
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = AveragePooling2D((7, 7), name='avg_pool', padding = 'same')(x)
    
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc10')(x)

    inputs = img_input
    
    model = Model(inputs, x, name='resnet50')  
  
    if compile_flag:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# # # # # # # # # #
#     alexnet     #
# # # # # # # # # #
def alexnet(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False):
    AlexNet_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(height, width, depth)),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)), # 原本是這行，但要改成下面那行才不會報錯
        keras.layers.MaxPool2D((5,5), padding='same'), 
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.MaxPool2D((5,5), padding='same'), 
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.MaxPool2D((5,5), padding='same'), 
        keras.layers.Flatten(),
        keras.layers.Dense(2048, activation='relu'), 
        keras.layers.Dropout(0.7), # 0.5
        keras.layers.Dense(2048, activation='relu'), 
        keras.layers.Dropout(0.4), # 0.5
        keras.layers.Dense(2048, activation='relu'), 
        keras.layers.Dropout(0.6), # 0.5
        keras.layers.Dense(2048, activation='relu'), 
        keras.layers.Dropout(0.5), # 0.5
        keras.layers.Dense(2048, activation='relu'), 
        keras.layers.Dropout(0.8), # 0.5
        keras.layers.Dense(classes, activation='softmax'), # classes是總共幾類
    ])
    
    if compile_flag:
        AlexNet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return AlexNet_model

# # # # # # # # # #
#      lenet      #
# # # # # # # # # #
def lenet(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False):
    # initialize the model
    LeNet_model = Sequential()
    
    # first layer, convolution and pooling
    LeNet_model.add(Conv2D(input_shape=(height, width, depth), kernel_size=(5, 5), filters=6, strides=(1,1), activation='tanh'))
    LeNet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # second layer, convolution and pooling
    LeNet_model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=16, strides=(1,1), activation='tanh'))
    LeNet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Fully connection layer
    LeNet_model.add(Flatten())
    LeNet_model.add(Dense(120, activation = 'tanh'))
    LeNet_model.add(Dense(84, activation = 'tanh'))
    
    # softmax classifier
    LeNet_model.add(Dense(classes))
    LeNet_model.add(Activation("softmax"))
    
    if compile_flag:
        LeNet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return LeNet_model

def model(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False):
     
    resnet50_model = resnet50(height, width, depth, classes, compile_flag)
    alexnet_model = alexnet(height, width, depth, classes, compile_flag)
    lenet_model = lenet(height, width, depth, classes, compile_flag)
    
    return resnet50_model, alexnet_model, lenet_model