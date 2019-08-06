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

        gs = [Generator(), Generator()]
        ds = [Discriminator(), Discriminator()]
        try:
            print("Started training")
            epochs_per_cycle = 10
            for i in range(1000):

                timestamp = f'step_{_globals.step:08d}'
                for g in gs:
                    g.sample(400, save_to=os.path.join(generated_dir, f"output_{g.id}", timestamp))

                for _ in range(epochs_per_cycle//5 + 1):
                    for d in ds:
                        for g in gs:
                            timestamp = f'step_{_globals.step:08d}'
                            g.save(timestamp), d.save(timestamp)
                            g.save('latest'), d.save('latest')
                            train(cifar_dataset, 1, g=g, d=d)

        except KeyboardInterrupt as e:
            print(e)
        finally:
            print("Saving the model")
            for g in gs:
                g.save("latest")
            for d in ds:
                d.save("latest")