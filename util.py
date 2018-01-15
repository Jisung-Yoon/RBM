import numpy as np
import matplotlib.pyplot as plt
import os


def check_and_make_dir(path='./result'):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
        return False


def reshaped_and_save_images(images):
    reshaped_images = np.reshape(images, (-1, 28, 28))
    f, axarr = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axarr[i, j].imshow(reshaped_images[i * 10 + j])
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

