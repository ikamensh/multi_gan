import tensorflow as tf

import numpy as np

def crop_resize_batch(x, min_size = 0.8):

    assert len(x.shape) == 4
    batch_size, w, h, colors = x.shape

    box_sizes_x = tf.random.uniform([batch_size, 1], min_size, 1.)
    box_sizes_y = tf.random.uniform([batch_size, 1], min_size, 1.)
    displacement_x = tf.random.uniform([batch_size, 1]) * (1 - box_sizes_x)
    displacement_y = tf.random.uniform([batch_size, 1]) * (1 - box_sizes_y)

    boxes = tf.concat([ displacement_x,
                        displacement_y,
                        displacement_x + box_sizes_x,
                        displacement_y + box_sizes_y],
                      1
                      )

    out = tf.image.crop_and_resize(x,
                             boxes=boxes,
                             box_indices=np.zeros(batch_size),
                             crop_size=(w, h))

    return out


def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation"""
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

if __name__ == "__main__":
    batch_size = 32
    tensor = np.random.uniform(-1, 1, size=[batch_size, 64, 64, 3])
    img = tf.constant(tensor)

    out = crop_resize_batch(img)
    print(out.shape)