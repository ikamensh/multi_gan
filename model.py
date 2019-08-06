import os
from tensorflow.python.keras import Sequential
import tensorflow as tf
from collections import defaultdict

class Model:

    check_dir: str
    net: Sequential
    optimizer: tf.keras.optimizers.Optimizer

    counters = defaultdict(int)

    def __init__(self):
        self.id = Model.counters[self.__class__]
        Model.counters[self.__class__] += 1

    def save(self, label:str):
        os.makedirs(self.check_dir, exist_ok=True)
        self.net.save(os.path.join(self.check_dir, f'{label}_{self.id}.h5'))

    def load(self, label:str):
        self.net.load_weights(os.path.join(self.check_dir, label + '.h5'))

    def update(self, tape: tf.GradientTape, loss):
        grad = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.net.trainable_variables))