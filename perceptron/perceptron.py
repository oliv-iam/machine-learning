import numpy as np

def predict(X, W, b):
    return (np.dot(X,W) + b >= 0).astype(int)

def train(eta, n, X, Y):
    '''
    eta: 0 < eta < 1 (learning rate)
    n: number of iterations
    X: n_examples x n_features (training vectors)
    Y: n_examples x 1 (target values)
    '''

    # initialize weights by normal distribution
    rgen = np.random.RandomState()
    W = rgen.normal(loc=0.0, scale=0.01, size=np.shape(X)[1])
    b = 0.0
    errors = []

    # train for n iterations
    for i in range(n):
        Yn = predict(X, W, b)
        Dw = eta * (Y-Yn)
        errors.append(int(np.sum(abs(Y-Yn))))
        W += np.dot(Dw, X)
        b += np.sum(Dw)

    return W, b, errors