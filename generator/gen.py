import tensorflow as tf
import os
from typing import Union

from generator.net import make_generator_model
from config import generated_dir, colors, latent_dim, n_classes
from model import Model
from util.visualize import save_images

from util.diff_aug import crop_resize_batch, color

bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class Generator(Model):
    check_dir = os.path.join(generated_dir, "checkpoints", "generator")

    def __init__(self, size_factor):
        self.net = make_generator_model(latent_dim,
                                        num_classes=n_classes,
                                        color_ch=colors,
                                        size_factor = size_factor)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        super().__init__()

    @staticmethod
    def loss(fake_output, class_preds, class_labels):
        return bin_cross_entropy(tf.ones_like(fake_output), fake_output) + \
               cross_entropy(class_labels, class_preds) / 3

    def forward(self, seed: tf.Tensor, cls: tf.Tensor, training: bool):
        generated_images = self.net([seed, cls], training)
        return crop_resize_batch(generated_images)



    def sample(self,
               seed: tf.Tensor = None,
               cls: Union[tf.Tensor, int] = None,
               n: int = None,
               save_to: str = None,
               training: bool = False):

        assert seed is None or n is None

        if n is None:
            if seed is not None:
                n = seed.shape[0]
            else:
                n = 1

        if seed is None:
            seed = tf.random.normal([n, latent_dim])

        if cls is None:
            class_target = tf.tile(tf.range(0, n_classes), tf.constant([1 + n // n_classes]))
            cls = class_target[:n]

        elif isinstance(cls, int):
            cls = tf.ones([n, 1], dtype=tf.int32) * cls
        else:
            cls = cls[:n]

        images = self.forward(seed, cls, training)
        if save_to:
            save_images(images, save_to)

        return images


if __name__ == "__main__":
    g = Generator()
    # g.load("latest")
    g.sample(cls=0, n=100, save_to=os.path.join(generated_dir, 'test_gen_1epoch'))
