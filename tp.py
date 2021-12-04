
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# get all paths of image
all_image_paths = glob.glob('sharks/*/*.jpg')
print(len(all_image_paths))
print(all_image_paths[:3])

# mess up the order
random.seed(0)
random.shuffle(all_image_paths)

# get all labels name (string)
all_labels_name = [img.split('/')[1] for img in all_image_paths]

# classifiacation
labels_name = np.unique(all_labels_name)
# print(labels_name)

# create two dictionair for search value
label_to_index = dict((label, index) for index, label in enumerate(labels_name))
index_to_label = dict((label, index) for label, index in enumerate(labels_name))
# print(label_to_index)
# print(index_to_label)

# get all labels index (int)
all_labels_index = [label_to_index.get(name) for name in all_labels_name]
print(all_labels_index)

# Separate data train and data validation
SIZE = len(all_image_paths)

# create a dataset
ds_train = tf.data.Dataset.from_tensor_slices((all_image_paths[:1200], all_labels_index[:1200]))
ds_val = tf.data.Dataset.from_tensor_slices((all_image_paths[1200:], all_labels_index[1200:]))

def caption_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image/255
    # print(image.shape)
    # print(image.dtype)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(caption_image, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.map(caption_image, num_parallel_calls=AUTOTUNE)
print(ds_train)
# definit batch siez
BATCH_SIZE = 32

# mess up the order, set the number of pictures in the buffer ######################### tester without shuffle line:15
ds_train = ds_train.repeat().shuffle(200).batch(BATCH_SIZE)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

ds_val = ds_val.batch(BATCH_SIZE)

# model
from tensorflow.keras.layers import Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(64,(3,3), activation="relu", input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(256,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(512,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(512,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(GlobalMaxPooling2D())

model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(14, activation="softmax"))

# model.summary()


model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=["accuracy"])

train_epochs = 1200//BATCH_SIZE
val_epochs = 275//BATCH_SIZE 


his = model.fit(
    ds_train,
    epochs=10, 
    steps_per_epoch=train_epochs,
    validation_data=ds_val, 
    validation_steps=val_epochs,
    verbose=1
)

