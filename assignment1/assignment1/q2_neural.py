#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    M = data.shape[0]

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)   # h: M X H
    z2 = np.dot(h, W2) + b2
    y_hat = softmax(z2)  # y_hat: M X Dy
    cost = - np.sum(labels * np.log(y_hat))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradz2 = y_hat - labels  # gradz2: M X Dy

    # gradW2 calculation
    z2gradw2 = np.zeros((H, M))  # z2gradw2: H X M
    for i in xrange(H):
        z2gradw2[i, :] = h[:, i]
    gradW2 = np.dot(z2gradw2, gradz2)
    # easier way:
    # gradW2 = np.dot(h.T, gradz2)

    z2gradb2 = np.ones((1, M))  # z2gradb2: 1 X M
    gradb2 = np.dot(z2gradb2, gradz2)  # gradb2: 1 X Dy
    # easier way:
    # gradb2 = np.sum(gradz2, axis=0, keepdims=True)

    gradh = np.dot(gradz2, W2.T)  # gradh:  M x H

    gradz1 = gradh * sigmoid_grad(h)  # gradz1: M X H

    z1gradw1 = np.zeros((Dx, M))  # z1gradw1: Dx X M
    for i in xrange(Dx):
        z1gradw1[i, :] = data[:, i]
    gradW1 = np.dot(z1gradw1, gradz1)  # gradW1: Dx X H
    # easier way:
    # gradW1 = np.dot(data.T, gradz1)

    z1gradb1 = np.ones((1, M))  # z1gradb1: 1 X M
    gradb1 = np.dot(z1gradb1, gradz1)  # gradb1: 1 X H
    # easier way:
    # gradb1 = np.sum(gradz1, axis=0, keepdims=True)

    assert (np.all(gradW2.shape == W2.shape))
    assert (np.all(gradb2.shape == b2.shape))
    assert (np.all(gradW1.shape == W1.shape))
    assert (np.all(gradb1.shape == b1.shape))
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
       gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 15]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
