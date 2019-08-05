import os

from matplotlib import pyplot as plt

from generator.generator import Generator
from config import generated_dir

dir_samples = os.path.join(generated_dir, "samples_human")
os.makedirs(dir_samples, exist_ok=True)


def generate_and_save_images(model: Generator, test_input, timestamp):
    predictions = model.net(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow( (predictions[i] * 127.5 + 127.5).numpy().astype('uint8') )
        plt.axis('off')

    plt.savefig(os.path.join(dir_samples, f'image_at_step_{timestamp:09d}.png'))
    plt.close(fig)