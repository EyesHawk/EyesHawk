import keras.datasets.mnist as mnist
import os
# import panda as pd


def get_mnist_datasets(res_dir):
    mnist_path = mnist.get_file(fname=res_dir
                                , origin='https://s3.amazonaws.com/img-datasets/mnist.npz'
                                , file_hash='8a61469f7ea1b51cbae51d4f78837e45'
                                )
    data = mnist.load_data(mnist_path)
    # x_train, y_train = data[0][0], data[0][1]
    # x_test, y_test = data[0][0], data[0][1]
    return data
