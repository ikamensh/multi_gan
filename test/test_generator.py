from generator import Generator
from generator.net import make_generator_model
import os

import imageio
import tensorflow as tf

import config


def test_generator_net():
    make_generator_model(100, 10)


def test_outputs_f32():
    g = Generator()

    img = g.sample()

    assert img.dtype == tf.float32

def test_sample_2(tmpdir):
    g = Generator()

    files_before = set(os.listdir(tmpdir))
    g.sample(cls=0, n=2, save_to=tmpdir)
    new_imgs = set(os.listdir(tmpdir)) - files_before

    assert len(new_imgs) == 2


def test_sample_valid_image(tmpdir):
    g = Generator()

    files_before = set(os.listdir(tmpdir))

    g.sample(cls=0, n=1, save_to=tmpdir)

    new_imgs = set(os.listdir(tmpdir)) - files_before

    assert len(new_imgs) == 1

    file = next(iter(new_imgs))
    path = os.path.join(tmpdir, file)
    img = imageio.imread(path)

    assert len(img.shape) == 3

def test_sample_classes(tmpdir):
    g = Generator()

    files_before = set(os.listdir(tmpdir))

    for cls in range(config.n_classes):
        g.sample(cls=cls, n=1, save_to=tmpdir)

    new_imgs = set(os.listdir(tmpdir)) - files_before

    assert len(new_imgs) == config.n_classes


def test_sample_diff_classes(tmpdir):
    g = Generator()

    files_before = set(os.listdir(tmpdir))

    g.sample(cls=tf.range(config.n_classes), n=config.n_classes, save_to=tmpdir)

    new_imgs = set(os.listdir(tmpdir)) - files_before

    assert len(new_imgs) == config.n_classes