from tensorflow.python.keras import layers
import tensorflow as tf

def make_discriminator_model(num_classes, color_ch = 3, size_factor = 4):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(16 * size_factor, 4, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2D(32 * size_factor, 5, strides=2, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())
    cnn.add(layers.Dropout(0.2))

    cnn.add(layers.Conv2D(16 * size_factor, 4, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Conv2D(32 * size_factor, 5, strides=2, padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())

    cnn.add(layers.Dropout(0.2))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(32 * size_factor))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.LeakyReLU())


    image = layers.Input(shape=(32, 32, color_ch))
    features = cnn(image)

    fake = layers.Dense(1, activation='sigmoid', name='generation')(features)
    aux = layers.Dense(num_classes, activation='softmax', name='auxiliary')(features)

    discriminator = tf.keras.Model(image, [fake, aux])

    assert discriminator.input_shape == (None, 32, 32, 3)

    return discriminator