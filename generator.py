from tensorflow.python.keras import layers
import tensorflow as tf

def _make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*64, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 64)))
    assert model.output_shape == (None, 7, 7, 64) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    def __init__(self):
        self.net = _make_generator_model()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    @staticmethod
    def loss(fake_output):
        return _cross_entropy(tf.ones_like(fake_output), fake_output)

    def update(self, tape: tf.GradientTape, loss):
        grad = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.net.trainable_variables))