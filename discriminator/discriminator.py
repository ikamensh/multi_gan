import tensorflow as tf
import os

from config import generated_dir, n_classes
from discriminator.net import make_discriminator_model
from model import Model

class GanMetrics:
    real_loss = 'discr_real_loss'
    fake_loss = 'discr_fake_loss'
    real_acc = 'discr_real_acc'
    fake_acc = 'discr_fake_acc'

_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Discriminator(Model):
    check_dir = os.path.join(generated_dir, "checkpoints", "discriminator")

    def __init__(self):
        self.net = make_discriminator_model(n_classes)
        self.optimizer = tf.keras.optimizers.Adam(3e-4)

        self.real_accuracy = tf.metrics.BinaryAccuracy()
        self.fake_accuracy = tf.metrics.BinaryAccuracy()
        self.real_loss = tf.metrics.Mean()
        self.fake_loss = tf.metrics.Mean()
        super().__init__()

    def loss(self, real_output, fake_output):
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        real_loss = _cross_entropy(real_labels, real_output)
        fake_loss = _cross_entropy(fake_labels, fake_output)

        self.real_accuracy.update_state(real_labels, real_output)
        self.fake_accuracy.update_state(fake_labels, fake_output)
        self.real_loss.update_state(real_loss)
        self.fake_loss.update_state(fake_loss)

        total_loss = real_loss + fake_loss
        return total_loss

    def log_metrics(self):
        tf.summary.scalar(GanMetrics.real_loss, self.real_loss.result(), self.optimizer.iterations)
        tf.summary.scalar(GanMetrics.fake_loss, self.fake_loss.result(), self.optimizer.iterations)

        tf.summary.scalar(GanMetrics.real_acc, self.real_accuracy.result(), self.optimizer.iterations)
        tf.summary.scalar(GanMetrics.fake_acc, self.fake_accuracy.result(), self.optimizer.iterations)

        self.real_accuracy.reset_states()
        self.fake_accuracy.reset_states()
        self.real_loss.reset_states()
        self.fake_loss.reset_states()




