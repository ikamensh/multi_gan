from tensorflow.python.keras import layers
import tensorflow as tf

def make_generator_model(noise_dim, num_classes, color_ch=3, size_factor = 4):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Dense(7*7*64 * size_factor, input_shape=(noise_dim,)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Dense(7*7*32 * size_factor, input_shape=(noise_dim,)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Reshape((7, 7, 32 * size_factor)))
    assert cnn.output_shape == (None, 7, 7, 32 * size_factor), cnn.output_shape

    cnn.add(layers.Conv2DTranspose(32 * size_factor, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 32 * size_factor), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(64 * size_factor, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 64 * size_factor), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(32* size_factor, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 32 * size_factor), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(64 * size_factor, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 7, 7, 64 * size_factor), cnn.output_shape

    cnn.add(layers.Conv2DTranspose(16 * size_factor, (4, 4), strides=(2, 2), padding='valid', use_bias=False))
    assert cnn.output_shape == (None, 16, 16, 16 * size_factor), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(64 * size_factor, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    assert cnn.output_shape == (None, 16, 16, 64 * size_factor), cnn.output_shape
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2DTranspose(color_ch, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    assert cnn.output_shape == (None, 32, 32, color_ch), cnn.output_shape

    latent = layers.Input(shape=(noise_dim, ))

    # this will be our label
    image_class = layers.Input(shape=(1,), dtype='int32')

    cls = layers.Flatten()(layers.Embedding(num_classes, noise_dim,
                              embeddings_initializer='glorot_normal')(image_class))

    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    # corrector = tf.keras.Sequential()
    # corrector.add(layers.Flatten(input_shape=(32,32,color_ch)))
    # corrector.add(layers.Dense(32*32*color_ch))
    # corrector.add(layers.Reshape((32,32,color_ch)))
    # summ = layers.Add()([corrector(fake_image), fake_image])
    # corrected = layers.Activation(activation='tanh') (summ)


    generator = tf.keras.Model([latent, image_class], fake_image)
    assert generator.output_shape == (None, 32, 32, color_ch), generator.output_shape

    return generator
