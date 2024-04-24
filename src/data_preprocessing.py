from aeon.datasets import load_classification
import numpy as np
from sklearn.model_selection import train_test_split

X, y = load_classification("GunPoint")
X=X.transpose((0,2,1))
y=np.array(y).astype('int')-1



X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)


m=X_train.mean()
std=X_train.std()

def standardize(X):
  return (X-m)/std

np.save("data/X_train.npy", standardize(X_train))
np.save("data/X_valid.npy", standardize(X_valid))
np.save("data/y_train.npy", y_train)
np.save("data/y_valid.npy", y_valid)

