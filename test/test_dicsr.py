from discriminator.net import make_discriminator_model
from discriminator import Discriminator

from config import n_classes


import numpy as np

def test_can_build():
    d = make_discriminator_model(n_classes)

def test_net():
    d = make_discriminator_model(n_classes)

    fake_img = np.ones([10,32,32,3], dtype=np.float64)

    is_fake, classes = d(fake_img)

    assert is_fake.shape == (10, 1)
    assert classes.shape == (10, n_classes)

def test_build_discr():
    d = Discriminator()