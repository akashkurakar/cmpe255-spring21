from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from mlxtend.data import loadlocal_mnist
import platform

np.random.seed(42)
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "./"

X, y = loadlocal_mnist(
    images_path='../train-images-idx3-ubyte',
    labels_path='../train-labels-idx1-ubyte')
np.savetxt(fname='images.csv',
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='labels.csv',
           X=y, delimiter=',', fmt='%d')

print(X.shape)

X_train, X_test, y_train, y_test = X[:
                                     60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Example: Binary number 4 Classifier
y_train_4 = (y_train == 4)
y_test_4 = (y_test == 4)
sgd_clf = SGDClassifier(random_state=42)  # instantiate
sgd_clf.fit(X_train, y_train_4)  # train the classifier


def save_fig(fig_id, tight_layout=True):
    path = "./Foo.png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def random_digit():
    some_digit = X[3652]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

    save_fig(some_digit)
    plt.show()


def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('../train-images-idx3-ubyte',
                             version=1, cache=True)
        # fetch_openml() returns targets as strings
        mnist.target = mnist.target.astype(np.int8)
        sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    print(mnist["data"]), print(mnist["target"])


def sort_by_target(mnist):
    reorder_train = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def train_predict(some_digit):

    # TODO
    # print prediction result of the given input some_digit

    #some_digit = X[3652]
    print(sgd_clf.predict([some_digit]))  # make it predict some digit


def calculate_cross_val_score():
    # TODO
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(sgd_clf, X_train, y_train_4, cv=3, scoring="accuracy"))


def test() -> None:
    random_digit()
    train_predict(X[3652])
    calculate_cross_val_score()


if __name__ == "__main__":
    test()
