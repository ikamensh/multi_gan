import os
import time
import tensorflow as tf

BUFFER_SIZE = 60000
BATCH_SIZE = 64
noise_dim = 100
num_examples_to_generate = 16

(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()


from util.score import inception_score
from util.imgs import gather_np

from ilya_ezplot import Metric, ez_plot

m = Metric('epoch', 'inc_score')

scan_dir = '../output'
for file in os.listdir(scan_dir):
    current = os.path.join(scan_dir, file)
    if os.path.isdir(current):
        print(file)
        epoch = int(file.replace('epoch_', ''))
        imgs = gather_np(current)
        inc_score = inception_score(imgs)
        m.add_record(epoch, inc_score)
        break

plots_folder = "plots"
ez_plot(m, plots_folder)
