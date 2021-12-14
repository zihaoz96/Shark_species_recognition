import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model.modelNet import modelNet
from model.model_v1 import model_v1
from model.VGG16 import VGG16
from model.DenseNet import DenseNet
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
import datetime

# get all paths of image
all_image_paths = glob.glob('sharks/*/*.jpg')
print(len(all_image_paths))
print(all_image_paths[:3])

# mess up the order
random.seed(0)
random.shuffle(all_image_paths)

# get all labels name (string) woking on linux terminal
all_labels_name = [img.split('/')[1] for img in all_image_paths]
# get all labels name (string) woking on windows powershell
# all_labels_name = [img.split('\\')[1] for img in all_image_paths]

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
# print(all_labels_index)

# Separate data train and data validation
SIZE = len(all_image_paths)

# create a dataset
ds_train = tf.data.Dataset.from_tensor_slices((all_image_paths[:1200], all_labels_index[:1200]))
ds_val = tf.data.Dataset.from_tensor_slices((all_image_paths[1200:], all_labels_index[1200:]))

def caption_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image/255
    # print(image.shape)
    # print(image.dtype)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(caption_image, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.map(caption_image, num_parallel_calls=AUTOTUNE)
print(ds_train)
# definit batch siez
BATCH_SIZE = 16

# mess up the order, set the number of pictures in the buffer ######################### tester without shuffle line:15
ds_train = ds_train.repeat().shuffle(10).batch(BATCH_SIZE)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

ds_val = ds_val.shuffle(10).batch(BATCH_SIZE)

# model
# model = modelNet(14)
# model = model_v1(14)
model = DenseNet(14)

model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    2e-5,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr_schedule),metrics=["accuracy"])

train_epochs = 1200//BATCH_SIZE
val_epochs = 315//BATCH_SIZE 


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
my_callbacks = [
    EarlyStopping(patience=2),
    ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, mode='max'),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
]

his = model.fit(
    ds_train,
    epochs=100, 
    validation_data=ds_val, 
    steps_per_epoch=train_epochs,
    validation_steps=val_epochs,
    verbose=1,
    callbacks=[my_callbacks],
    shuffle=True
)

fig = plt.figure(figsize=(6, 3), dpi=150)
plt.plot(his.epoch, his.history.get('loss'), label = 'loss')
plt.plot(his.epoch, his.history.get('val_loss'), label = 'val_loss')
plt.legend()
plt.savefig("fig/loss_fig")

plt.clf()
fig = plt.figure(figsize=(6, 3), dpi=150)
plt.plot(his.epoch, his.history.get('accuracy'), label = 'accuracy')
plt.plot(his.epoch, his.history.get('val_accuracy'), label = 'val_accuracy')
plt.legend()
plt.savefig("fig/accuracy_fig")

