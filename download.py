'''
Downloads: MNIST dataset
'''

from __future__ import print_function
import os
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Download MNIST data sets')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['mnist'],
                    help='name of dataset to download [mnist]')


# Functions for Downloading MNIST Dataset
def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if check_and_make_dir(data_dir):
        print('Found MNIST - skip')
        return

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_name_list = ['train-images-idx3-ubyte.gz',
                      'train-labels-idx1-ubyte.gz',
                      't10k-images-idx3-ubyte.gz',
                      't10k-labels-idx1-ubyte.gz']

    for file_name in file_name_list:
        url = (base_url + file_name).format(**locals())
        print(url)
        download_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', download_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', download_path]
        print('Unzip ', file_name)
        subprocess.call(cmd)


def check_and_make_dir(path='./data'):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
        return False


if __name__ == '__main__':
    args = parser.parse_args()
    check_and_make_dir('./data')

    if 'mnist' in args.datasets:
        download_mnist('./data')
