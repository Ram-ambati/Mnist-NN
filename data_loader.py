import numpy as np
import pandas as pd


def one_hot(labels, num_classes=10):
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1
    return out


def load_mnist_csv(path):
    data = pd.read_csv(path)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values / 255.0
    labels = one_hot(labels)
    return images, labels


def get_batch(images, labels, batch_size):
    idx = np.random.choice(len(images), batch_size, replace=False)
    return images[idx], labels[idx]
