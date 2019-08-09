from util.cifar import cifar_dataset

def test_has_labels():

    for imgs, labels in cifar_dataset:
        assert imgs.shape[1:] == (32, 32, 3,)
        assert labels.shape[1:] == (1,)