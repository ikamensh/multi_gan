from ilya_ezplot.plot.plot import plt
import os
import time
import tensorflow as tf

BUFFER_SIZE = 60000
BATCH_SIZE = 64
noise_dim = 100
num_examples_to_generate = 16

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

from generator import Generator
from discriminator import Discriminator

g = Generator()
d = Discriminator()


@tf.function
def train_step(images, discr: Discriminator, gen: Generator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen.net(noise, training=True)

        real_output = discr.net(images, training=True)
        fake_output = discr.net(generated_images, training=True)

        gen_loss = gen.loss(fake_output)
        disc_loss = discr.loss(real_output, fake_output)

    discr.update(disc_tape, disc_loss)
    gen.update(gen_tape, gen_loss)

def train(dataset, epochs):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, d, g)

        generate_and_save_images(g, epoch + 1, seed)
        print (f'Time for epoch {(epoch + 1)} is {(time.time() - start):.2f} sec')

    generate_and_save_images(g, epochs, seed)


os.makedirs('output', exist_ok=True)
def generate_and_save_images(model:Generator, epoch:int, test_input):
    predictions = model.net(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join("output", f'image_at_epoch_{epoch:04d}.png') )

if __name__ == "__main__":
    train(train_dataset, 5)
