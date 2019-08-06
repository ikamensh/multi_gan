from tensorflow.python.keras import layers
import tensorflow as tf

def make_generator_model(noise_dim, num_classes, color_ch=3):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Dense(7*7*128, use_bias=False, input_shape=(noise_dim,)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Reshape((7, 7, 128)))
    assert cnn.output_shape == (None, 7, 7, 128), cnn.output_shape

    cnn.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 128), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(256, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 256), cnn.output_shape

    cnn.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='valid', use_bias=False))
    assert cnn.output_shape == (None, 16, 16, 64), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(color_ch, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert cnn.output_shape == (None, 32, 32, color_ch), cnn.output_shape

    latent = layers.Input(shape=(noise_dim, ))

    # this will be our label
    image_class = layers.Input(shape=(1,), dtype='int32')

    cls = layers.Flatten()(layers.Embedding(num_classes, noise_dim,
                              embeddings_initializer='glorot_normal')(image_class))

    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    generator = tf.keras.Model([latent, image_class], fake_image)
    assert generator.output_shape == (None, 32, 32, color_ch), generator.output_shape

    return generator
