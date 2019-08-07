import tensorflow as tf
import os

from config import generated_dir, n_classes
from discriminator.net import make_discriminator_model
from model import Model
import _globals

class GanMetrics:
    real_loss = 'discr_real_loss'
    fake_loss = 'discr_fake_loss'
    real_acc = 'discr_real_acc'
    fake_acc = 'discr_fake_acc'

    aux_loss = 'discr_aux_loss'


bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class Discriminator(Model):
    check_dir = os.path.join(generated_dir, "checkpoints", "discriminator")

    def __init__(self, size_factor):
        self.net = make_discriminator_model(n_classes, size_factor=size_factor)
        self.lr = tf.Variable(3e-5)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.real_accuracy = tf.metrics.BinaryAccuracy()
        self.fake_accuracy = tf.metrics.BinaryAccuracy()
        self.real_loss = tf.metrics.Mean()
        self.fake_loss = tf.metrics.Mean()
        self.aux_loss = tf.metrics.Mean()

        super().__init__()

    def loss(self, real_output, fake_output, class_pred, labels):
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        real_loss = bin_cross_entropy(real_labels, real_output)
        fake_loss = bin_cross_entropy(fake_labels, fake_output)

        class_loss = cross_entropy(labels, class_pred)

        self.real_accuracy.update_state(real_labels, real_output)
        self.fake_accuracy.update_state(fake_labels, fake_output)
        self.real_loss.update_state(real_loss)
        self.fake_loss.update_state(fake_loss)
        self.aux_loss.update_state(class_loss)

        total_loss = 2 * real_loss + fake_loss + class_loss
        return total_loss

    def log_metrics(self):
        tf.summary.scalar(GanMetrics.real_loss, self.real_loss.result(), _globals.step)
        tf.summary.scalar(GanMetrics.fake_loss, self.fake_loss.result(), _globals.step)

        tf.summary.scalar(GanMetrics.real_acc, self.real_accuracy.result(), _globals.step)

        fake_acc = self.fake_accuracy.result()
        if fake_acc > 0.8:
            self.lr.assign(1e-6)
        elif fake_acc < 0.2:
            self.lr.assign(2e-4)
        else:
            self.lr.assign(3e-5)

        tf.summary.scalar(GanMetrics.fake_acc, self.fake_accuracy.result(), _globals.step)
        tf.summary.scalar(GanMetrics.aux_loss, self.aux_loss.result(), _globals.step)
        tf.summary.scalar('discr learning rate', self.lr.value(), _globals.step)

        self.real_accuracy.reset_states()
        self.fake_accuracy.reset_states()
        self.real_loss.reset_states()
        self.fake_loss.reset_states()
        self.aux_loss.reset_states()




