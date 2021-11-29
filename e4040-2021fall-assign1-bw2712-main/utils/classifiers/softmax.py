import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
      This adjusts the weights to minimize loss.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    N = X.shape[0]
    C = W.shape[1]
    epsilon = 1e-15
    
    for i in range(N):
        f = np.dot(X[i], W)
        f = f - np.max(f) + epsilon
        s = np.sum(np.exp(f)) + epsilon
        softmax = np.exp(f) / s
        
        loss = loss - np.log(softmax[y[i]])
        
        for j in range(C):
            dW[:, j] += X[i] * softmax[j]
        dW[:, y[i]] -= X[i]
    loss = loss / N 
    loss = loss + reg * np.sum(W ** 2)
    dW = dW / N + 2 * reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    N = X.shape[0]
    epsilon = 1e-15
    
    f = np.dot(X, W)
    f = f - np.max(f, axis = 1, keepdims=True) + epsilon
    s = np.sum(np.exp(f), axis = 1, keepdims=True) + epsilon
    softmax = np.exp(f) / s
    loss = np.sum(-1*np.log(softmax[range(N), y]))

    softmax[range(N), y] -= 1
    dW = np.dot(X.T, softmax)

    loss = loss / N + reg * np.sum(W ** 2)
    dW = dW / N + 2 * reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
