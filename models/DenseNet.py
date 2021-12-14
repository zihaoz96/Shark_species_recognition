
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121

def DenseNet(nbr_class):
    base_model = DenseNet121(weights="imagenet", include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(14, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(nbr_class, activation="softmax"))

    return model
