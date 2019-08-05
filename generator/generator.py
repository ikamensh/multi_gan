import tensorflow as tf
import imageio
import os

from config import generated_dir
from generator.net import make_generator_model

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    check_dir = os.path.join(generated_dir, "checkpoints", "generator")
    def __init__(self):
        self.net = make_generator_model()
        self.optimizer = tf.keras.optimizers.Adam(3e-4)

    @staticmethod
    def loss(fake_output):
        return _cross_entropy(tf.ones_like(fake_output), fake_output)

    def update(self, tape: tf.GradientTape, loss):
        grad = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.net.trainable_variables))

    def sample(self, n = 1, save_to: str = None):
        noise = tf.random.normal([n, 100])
        generated_images = self.net(noise, training=False)
        if save_to:

            os.makedirs(save_to, exist_ok=True)
            normalized = (generated_images * 127.5 + 127.5).numpy().astype('uint8')
            for i, img in enumerate(normalized):
                imageio.imsave( os.path.join(save_to, f"{i+1}.jpg"), img)

        return generated_images

    def save(self, label:str):
        os.makedirs(self.check_dir, exist_ok=True)
        self.net.save(os.path.join(self.check_dir, label + '.h5'))

    def load(self, label:str):
        self.net.load_weights(os.path.join(self.check_dir, label + '.h5'))


if __name__ == "__main__":
    g = Generator()
    # g.load("latest")
    g.sample(100, os.path.join(generated_dir, 'test_gen_1epoch'))



