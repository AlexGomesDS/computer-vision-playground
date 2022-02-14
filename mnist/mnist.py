# methods to import and preprocess MNIST data as well as visualize it
import os

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def plot_mnist(x, y, n=10):
    """
    Plots the first n images of the MNIST dataset.
    """
    plt.figure(figsize=(13, 3))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"Label: {y[i]}")

    plt.show()


class Dataset:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.data, self.labels = self.load()
        self.classes = np.unique(self.labels)

    def load(self):
        """
        Loads the MNIST dataset from the path specified. If the path does not exist,
        load it from openml.org.
        """
        if self.dataset_path and os.path.exists(self.dataset_path):
            data_dict = np.load(self.dataset_path)
            x = data_dict["x"]  # array of integers in the range [0, 255]
            y = data_dict["y"]  # array of integers in the range [0, 9]

        else:
            # download the dataset from openml.org
            x, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True, as_frame=False)
            x, y = x.astype(np.float64) / 255., y.astype(np.int32)

            if self.dataset_path and not os.path.isfile(self.dataset_path):
                np.savez_compressed(self.dataset_path, x=x, y=y)

        return x, y

    def sample(self, n=100):
        """
        Returns a random sample of n images from the dataset.
        """
        indices = np.random.choice(np.arange(self.data.shape[0]), n, replace=False)
        return self.data[indices], self.labels[indices]

    def plot_instance(self, i):
        arr_pixel = self.data[i, :].reshape((28, 28))

        plot_mnist([arr_pixel], [self.labels[i]], 1)


def generate_train_test(ds, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        ds.data, ds.labels, test_size=test_size,
        random_state=random_state, stratify=ds.labels)

    np.savez_compressed("./data/mnist_train.npz", x=x_train, y=y_train)
    np.savez_compressed("./data/mnist_test.npz", x=x_test, y=y_test)
