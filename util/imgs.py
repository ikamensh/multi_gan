import os

import imageio
import numpy as np


def gather_np(folder):
    assert os.path.isdir(folder)
    result = []

    for file in os.listdir(folder):
        try:
            img = imageio.imread(os.path.join(folder, file))
            result.append(np.array(img))
        except Exception as e:
            print(f"Failed to read {file} as an image file:", e)

    return result

if __name__ == "__main__":


    # print(len(lst))
    # print(lst[0].shape)
    # print(set([type(e) for e in lst]))

    from util.score import inception_score

    mydir = "test_gen"
    lst = gather_np(mydir)
    print("before training:", inception_score(lst))


    mydir = "test_gen_1epoch"
    lst = gather_np(mydir)
    print("after training:", inception_score(lst))



