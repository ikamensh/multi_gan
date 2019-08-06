import os

import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
from train import train
from util.augment import cifar_dataset
from config import generated_dir
import _globals

if __name__ == "__main__":

    summary_writer = tf.summary.create_file_writer(os.path.join(generated_dir, 'summaries'))
    with summary_writer.as_default():

        g = Generator()
        d = Discriminator()
        try:
            print("Started training")
            epochs_per_cycle = 10
            for i in range(1000):
                timestamp = f'step_{_globals.step:08d}'
                g.save(timestamp), d.save(timestamp)
                g.save('latest'), d.save('latest')

                g.sample(400, save_to=os.path.join(generated_dir, "output", timestamp))
                train(cifar_dataset, epochs_per_cycle, g=g, d=d)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print("Saving the model")
            g.save("latest"), d.save("latest")