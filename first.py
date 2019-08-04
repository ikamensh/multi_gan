from ilya_ezplot.plot.plot import plt
import os
import time
import tensorflow as tf

BUFFER_SIZE = 60000
BATCH_SIZE = 64
noise_dim = 100
num_examples_to_generate = 16

global_epoch = 1

(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

from generator import Generator
from discriminator import Discriminator


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

def train(dataset, epochs, *, g, d):
    global global_epoch
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, d, g)

        generate_and_save_images(g, seed)
        print (f'Time for epoch {global_epoch} is {(time.time() - start):.2f} sec')
        global_epoch += 1

    generate_and_save_images(g, seed)


os.makedirs('output', exist_ok=True)
def generate_and_save_images(model:Generator, test_input):
    predictions = model.net(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig(os.path.join("output", f'image_at_epoch_{global_epoch:04d}.png') )
    plt.close(fig)


if __name__ == "__main__":

    summary_writer = tf.summary.create_file_writer('/tmp/summaries')
    with summary_writer.as_default():

        g = Generator()
        d = Discriminator()
        try:
            print("Started training")
            cycle = 5
            for i in range(1000):
                timestamp = f'epoch_{i*cycle}'
                g.save(timestamp), d.save(timestamp)
                g.save('latest'), d.save('latest')

                g.sample(500, save_to=os.path.join("output", timestamp))
                train(train_dataset, cycle, g=g, d=d)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print("Saving the model")
            g.save("latest"), d.save("latest")
