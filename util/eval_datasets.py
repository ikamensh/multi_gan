import os
import time
import tensorflow as tf

BUFFER_SIZE = 60000
BATCH_SIZE = 64
noise_dim = 100
num_examples_to_generate = 16

(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()


from util.score import inception_score

import random


duplicates = [img for img in train_images[:50]*2]
random.shuffle(duplicates)

import time

t = time.time()
print( inception_score([img for img in train_images[:50]]) )
print( inception_score([img for img in train_images[:50]*2]) )
print( inception_score( duplicates) )

print(time.time() - t)