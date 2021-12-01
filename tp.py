
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get all paths of image
all_image_paths = glob.glob('sharks/*/*.jpg')
print(len(all_image_paths))

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
# print(all_labels_index)

# Separate data train and data validation
SIZE = len(all_image_paths)

# create a dataset
ds_train = tf.data.Dataset.from_tensor_slices((all_image_paths[:SIZE*0.8], all_labels_index[:SIZE*0.8]))
ds_val = tf.data.Dataset.from_tensor_slices((all_image_paths[SIZE*0.8:], all_labels_index[SIZE*0.8:]))

def caption_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [256, 256])
    image = image/255
    # print(image.shape)
    # print(image.dtype)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_label_ds_train = ds_train.map(caption_image, num_parallel_calls=AUTOTUNE)
image_label_ds_val = ds_val.map(caption_image, num_parallel_calls=AUTOTUNE)
print(image_label_ds_train)

# definit batch siez
BATCH_SIZE = 32

# mess up the order, set the number of pictures in the buffer ######################### tester without shuffle line:15
image_label_ds_train = image_label_ds_train.repeat().shuffle(200).batch(BATCH_SIZE)
print(image_label_ds_train)









