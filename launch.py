import os

import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
from train import train, train_dataset
from config import generated_dir

if __name__ == "__main__":

    summary_writer = tf.summary.create_file_writer(os.path.join(generated_dir, 'summaries'))
    with summary_writer.as_default():

        g = Generator()
        d = Discriminator()
        try:
            print("Started training")
            cycle = 10
            for i in range(1000):
                timestamp = f'step_{d.step}'
                g.save(timestamp), d.save(timestamp)
                g.save('latest'), d.save('latest')

                g.sample(500, save_to=os.path.join(generated_dir, "output", f'{timestamp:08d}'))
                train(train_dataset, cycle, g=g, d=d)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print("Saving the model")
            g.save("latest"), d.save("latest")