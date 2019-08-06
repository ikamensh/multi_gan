import os
import time
import tensorflow as tf

from util.visualize import generate_and_save_images

from generator import Generator, noise_dim
from discriminator import Discriminator
import _globals
from util.cifar import BATCH_SIZE




def _train_step(images, discr: Discriminator, gen: Generator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen.net(noise, training=True)

        real_output = discr.net(images, training=True)
        fake_output = discr.net(generated_images, training=True)

        gen_loss = gen.loss(fake_output)
        disc_loss = discr.loss(real_output, fake_output)

    discr.update(disc_tape, disc_loss)
    gen.update(gen_tape, gen_loss)

from collections import defaultdict
_train_functions = defaultdict(lambda :tf.function(_train_step))

def train_step(images, discr: Discriminator, gen: Generator):
    train_function = _train_functions[(discr, gen)]
    train_function(images, discr, gen)


num_examples_to_generate = 16
def train(dataset, epochs, *, g: Generator, d: Discriminator):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    for epoch in range(epochs):
        start = time.time()
        generate_and_save_images(g, seed, _globals.step)
        for image_batch in dataset:
            train_step(image_batch, d, g)
            _globals.step += 1
            if not _globals.step % 100:
                d.log_metrics()

        print(f'Time for epoch {_globals.global_epoch} is {(time.time() - start):.2f} sec')
        _globals.global_epoch += 1

    generate_and_save_images(g, seed, _globals.step)




