import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


import platform

if 'Darwin' in str(platform.platform()):
    root = os.path.dirname(__file__)
else:
    root = 'E:'


generated_dir = os.path.join(root, "generated_mid_models", 'test_many')

latent_dim = 100
colors = 3


# from util.augment import unique_classes
# n_classes = len(unique_classes)
n_classes = 10

#
# def new_gen_dir():
#     global generated_dir
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     generated_dir = os.path.join('E:', "generated_mid_models", timestr)
