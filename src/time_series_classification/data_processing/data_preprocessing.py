import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


def standardize(X: np.array, m: float, std: float) -> float:
    """
    Standardize the data to have zero mean and unit variance

    Args:
        X (np.array)
        m (float) :mean
        std (float) :standard deviation

    Returns:
        float: standardized array
    """
    return (X - m) / std


def load_data():
    X_train = np.load("data/X_train.npy")
    X_valid = np.load("data/X_valid.npy")
    y_train = np.load("data/y_train.npy")
    y_valid = np.load("data/y_valid.npy")
    return X_train, X_valid, y_train, y_valid


def save_data(
    X: np.array,
    y: np.array,
    X_train: np.array,
    X_valid: np.array,
    y_train: np.array,
    y_valid: np.array,
):
    m = X_train.mean()
    std = X_train.std()
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    np.save("data/X_train.npy", standardize(X_train, m, std))
    np.save("data/X_valid.npy", standardize(X_valid, m, std))
    np.save("data/y_train.npy", y_train)
    np.save("data/y_valid.npy", y_valid)


if __name__ == "__main__":
    X, y = load_classification("ECG5000")
    X = X.transpose((0, 2, 1))
    y = np.array(y).astype("int") - 1

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)

    save_data(X, y, X_train, X_valid, y_train, y_valid)
