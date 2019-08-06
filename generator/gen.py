import tensorflow as tf
import os
from typing import Union

from generator.net import make_generator_model
from config import generated_dir, colors, latent_dim, n_classes
from model import Model
from util.visualize import save_images

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator(Model):
    check_dir = os.path.join(generated_dir, "checkpoints", "generator")
    def __init__(self):
        self.net = make_generator_model(latent_dim,
                                        num_classes=n_classes,
                                        color_ch=colors)
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        super().__init__()

    @staticmethod
    def loss(fake_output):
        return _cross_entropy(tf.ones_like(fake_output), fake_output)

    def sample(self, cls: Union[tf.Tensor, int], n = 1, save_to: str = None):
        noise = tf.random.normal([n, latent_dim])
        if isinstance(cls, int):
            cls = tf.ones([n, 1], dtype=tf.int32) * cls

        generated_images = self.net([noise, cls], training=False)

        if save_to:
            save_images(generated_images, save_to)

        return generated_images



if __name__ == "__main__":
    g = Generator()
    # g.load("latest")
    g.sample(cls=0, n=100, save_to=os.path.join(generated_dir, 'test_gen_1epoch'))



