# # # # # # # # # #
#      module     #
# # # # # # # # # #
import keras
import keras.backend as K
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import add, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Input, ZeroPadding2D, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam

# # # # # # # # # #
#     resnet50    #
# # # # # # # # # #
def resnet50(height = 224, width = 224, depth = 3, classes = 1):

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
  
    return model

def vgg19(width, height, depth, classes):
    
    model = Sequential()
    
    # padding = 'vaild' 則會直接捨去 padding 不足的區塊
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(width,height,depth),padding='same',activation='relu')) 
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    
    return model
