import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    h = 1 / (1 + np.exp(-x))
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################

    N = X.shape[0]
    D = X.shape[1]
    f = np.zeros([N])
    for i in range(N):
        for j in range(D):
            f[i] += X[i, j] * W[j]
        f[i] = sigmoid(f[i])
    for i in range(N):
        loss += -1 * (y[i] * np.log(f[i]) + (1-y[i]) * np.log(1-f[i]))
    loss /= N
    loss += reg * np.sum(W ** 2)

    dW = np.zeros([D, 1])
    for i in range(D):
        for j in range(N):
            dW[i] += X.T[i, j] * (f[j] - y[j])
        dW[i] = dW[i] / N
        
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    
    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    N = X.shape[0]
    y = y.reshape(-1,1)
    
    f = sigmoid(np.dot(X, W))
    
    cost = -(np.dot(y.T,np.log(f))+np.dot(1-y.T,np.log(1-f)))
    loss = np.sum(cost)
    loss /= N
    loss += reg * np.sum(W**2)
    dW = np.dot(X.T, f-y)
    dW /= N
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
