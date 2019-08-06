from util.cifar import cifar_dataset, BATCH_SIZE

def test_has_labels():

    for imgs, labels in cifar_dataset:
        assert imgs.shape == (BATCH_SIZE, 32, 32, 3)
        assert labels.shape == (BATCH_SIZE, 1)