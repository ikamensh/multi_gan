import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import platform

if 'Darwin' in str(platform.platform()):
    root = 'generated'
else:
    root = 'E:'


generated_dir = os.path.join(root, "acgan", timestr)
print(f'Using {generated_dir}')

latent_dim = 100
colors = 3
size_factor = 2


# from util.augment import unique_classes
# n_classes = len(unique_classes)
n_classes = 10
