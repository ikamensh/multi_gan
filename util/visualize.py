from __future__ import annotations
import os

from matplotlib import pyplot as plt
import imageio
from config import generated_dir


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from generator.gen import Generator


dir_samples = os.path.join(generated_dir, "samples_human")
os.makedirs(dir_samples, exist_ok=True)

def generate_and_save_images(model: Generator, test_input, timestamp):
    predictions = model.sample(seed=test_input)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow( (predictions[i] * 127.5 + 127.5).numpy().astype('uint8') )
        plt.axis('off')

    plt.savefig(os.path.join(dir_samples, f'image_at_step_{timestamp:09d}.png'))
    plt.close(fig)

def save_images(generated_images, save_to):
    os.makedirs(save_to, exist_ok=True)
    normalized = (generated_images * 127.5 + 127.5).numpy().astype('uint8')
    existing_files = os.listdir(save_to)
    to_idx = lambda s: int(s.split('.')[0])
    try:
        biggest_idx = max(to_idx(f) for f in existing_files) + 1
    except Exception:
        biggest_idx = 1
    for i, img in enumerate(normalized):
        imageio.imsave(os.path.join(save_to, f"{i + biggest_idx}.jpg"), img)