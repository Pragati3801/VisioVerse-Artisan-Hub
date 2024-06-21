import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import pathlib
dataset_url = 'C:\\Users\sinha\\OneDrive\\VisioVerse\All_pics'
data_dir = tf.keras.utils.get_file('All_pics', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

print(data_dir)