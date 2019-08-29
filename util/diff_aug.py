import tensorflow as tf

import numpy as np

# def crop_resize_batch(x, min_size = 0.8):
#
#     assert len(x.shape) == 4
#     batch_size, w, h, colors = x.shape
#
#     box_sizes_x = tf.random.uniform([batch_size, 1], min_size, 1.)
#     box_sizes_y = tf.random.uniform([batch_size, 1], min_size, 1.)
#     displacement_x = tf.random.uniform([batch_size, 1]) * (1 - box_sizes_x)
#     displacement_y = tf.random.uniform([batch_size, 1]) * (1 - box_sizes_y)
#
#     boxes = tf.concat([ displacement_x,
#                         displacement_y,
#                         displacement_x + box_sizes_x,
#                         displacement_y + box_sizes_y],
#                       1
#                       )
#
#     box_index =
#
#     out = tf.image.crop_and_resize(x,
#                              boxes=boxes,
#                              box_indices=tf.range(batch_size),
#                              crop_size=(w, h))
#
#     return out

def crop_same(x, min_size = 0.2, max_size = 0.5):

    assert len(x.shape) == 4
    batch_size, w, h, colors = x.shape

    box_sizes_x = tf.random.uniform([1, 1], min_size, max_size)
    box_sizes_y = tf.random.uniform([1, 1], min_size, max_size)
    displacement_x = tf.random.uniform([1, 1]) * (1 - box_sizes_x)
    displacement_y = tf.random.uniform([1, 1]) * (1 - box_sizes_y)

    box = tf.concat([ displacement_x,
                        displacement_y,
                        displacement_x + box_sizes_x,
                        displacement_y + box_sizes_y],
                      1
                      )

    false_idx = tf.range(batch_size)
    out = tf.image.crop_and_resize(x,
                             boxes=box,
                             box_indices=tf.zeros_like(false_idx),
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

    out = crop_same(img)
    print(out.shape)