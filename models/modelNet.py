
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model


def modelNet(nbr_class):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)
    mobile_net.trainable=False
  
    model = Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(nbr_class, activation = 'softmax')])

    return model