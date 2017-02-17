import numpy as np
#  ------------------------------ Dropout -----------------------------------

p = 0.5  # probability of keeping a unit active. higher = less dropout

X = np.random.randn(3, 1)
W1 = np.random.randn(5, 3)
W2 = np.random.randn(5, 5)
W3 = np.random.randn(1, 5)
b1 = np.random.randn(5, 1)
b2 = np.random.randn(5, 1)
b3 = np.random.randn(1, 1)


def train_step():
    """ X contains the data """

    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p  # first dropout mask
    H1 *= U1  # drop!
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p  # second dropout mask
    H2 *= U2  # drop!
    out = np.dot(W3, H2) + b3

    return out
    # backward pass: compute gradients... (not shown)
    # perform parameter update... (not shown)


def predict(X):
    # ensembled forward pass
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p  # NOTE: scale the activations
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p  # NOTE: scale the activations
    out = np.dot(W3, H2) + b3


train_step()