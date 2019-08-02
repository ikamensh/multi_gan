from tensorflow.python.keras import layers
import tensorflow as tf
import imageio
import os

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

    model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='valid', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3), model.output_shape

    return model

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    check_dir = "checkpoints"
    def __init__(self):
        self.net = _make_generator_model()
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
            for i, img in enumerate(generated_images.numpy()):
                imageio.imsave( os.path.join(save_to, f"{i+1}.jpg"), img)

        return generated_images

    def save(self, label:str):
        os.makedirs(self.check_dir, exist_ok=True)
        self.net.save(os.path.join(self.check_dir, label + '.h5'))

    def load(self, label:str):
        self.net.load_weights(os.path.join(self.check_dir, label + '.h5'))


if __name__ == "__main__":
    g = Generator()
    g.load("the_best")
    g.sample(100, 'test_gen_1epoch')



