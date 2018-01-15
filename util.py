import numpy as np
import matplotlib.pyplot as plt
import os


def check_and_make_dir(path='./result'):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
        return False


# Function to load mnist datasets
def load_mnist_datasets(seed):
    data_dir = os.path.join("./data", 'mnist')

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    training_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    training_labels = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float)

    training_labels = np.asarray(training_labels)
    test_labels = np.asarray(test_labels)

    images = np.concatenate((training_images, test_images), axis=0)
    images = np.reshape(images, [-1, 28 * 28])
    labels = np.concatenate((training_labels, test_labels), axis=0).astype(np.int)

    # shuffle data with given seed
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # Make label one_hot
    labels_one_hot = np.zeros((len(labels), 10), dtype=np.float)
    index_offset = np.arange(len(labels)) * 10
    labels_one_hot.flat[index_offset + labels.ravel()] = 1

    return images/255., labels_one_hot


# reshaped images and save images
def reshaped_and_save_images(images, path, epoch):
    reshaped_images = np.reshape(images, (-1, 28, 28))
    f, axarr = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axarr[i, j].imshow(reshaped_images[i * 10 + j])
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path + '/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')


