import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Input, MaxPooling2D
from tensorflow.keras import Model


def VGG16(nbr_class):
    # 224 224 3
    img_input = Input(shape=(224,224,3))

    # first convolution
    x = Conv2D(64, (3,3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), strides = (2,2))(x)

    # second convolution
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), strides = (2,2))(x)

    # third convolution
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), strides = (2,2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(nbr_class, activation='softmax')(x)


    return Model(img_input, x, name="vgg16")