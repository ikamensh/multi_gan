import time
import tensorflow as tf

from util.visualize import generate_and_save_images

from generator import Generator
from discriminator import Discriminator
import _globals
from util.cifar import BATCH_SIZE
from config import n_classes, latent_dim


# @tf.function
def train_step(images, labels, discr: Discriminator, gen: Generator):

    noise = tf.random.normal([BATCH_SIZE, latent_dim])
    class_target = tf.tile(tf.range(0, n_classes), tf.constant([1 + BATCH_SIZE//n_classes]) )
    class_target = class_target[:BATCH_SIZE]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen.forward(noise, class_target, training=True)

        real_output, real_classes = discr.net(images, training=True)
        fake_output, fake_classes = discr.net(generated_images, training=True)

        gen_loss = gen.loss(fake_output, class_preds=fake_classes, class_labels=class_target)
        disc_loss = discr.loss(real_output, fake_output, real_classes, labels)

    discr.update(disc_tape, disc_loss)
    gen.update(gen_tape, gen_loss)


num_examples_to_generate = 16
def train(dataset, epochs, *, g: Generator, d: Discriminator):
    seed = tf.random.normal([num_examples_to_generate, latent_dim])
    for epoch in range(epochs):
        start = time.time()
        generate_and_save_images(g, seed, _globals.step)
        for image_batch, labels in dataset:
            train_step(image_batch, labels, d, g)
            _globals.step += 1
            if not _globals.step % 100:
                d.log_metrics()

        print(f'Time for epoch {_globals.global_epoch} is {(time.time() - start):.2f} sec')
        _globals.global_epoch += 1

    generate_and_save_images(g, seed, _globals.step)




