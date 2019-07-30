from tensorflow.python.keras import layers
import tensorflow as tf

def _make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Discriminator:
    def __init__(self):
        self.net = _make_discriminator_model()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    @staticmethod
    def loss(real_output, fake_output):
        real_loss = _cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = _cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def update(self, tape: tf.GradientTape, loss):
        grad = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.net.trainable_variables))
