from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP (Multilayer Perceptrons) with an arbitrary number of dense hidden layers, and a softmax loss function. 
    For a network with L layers, the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        Do regularization for better model generalization.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        
        for layer in layers:
            X = layer.feedforward(X)
        loss, dx = softmax_loss(X, y)
        
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        
        for layer in reversed(layers):
            dx = layer.backward(dx)
        
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        
        weight_sum = 0
        for layer in layers:
            weight_sum += np.sum(layer.params[0]*layer.params[0])
        loss += self.reg*weight_sum

        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        Set learning rate to 0.00001.
        """
        ####################################################
        # TODO: Use SGD to update variables in layers.     #
        ####################################################
        ####################################################
        #           START OF YOUR CODE                     #
        ####################################################
        
        num_layers = self.num_layers
        layers = self.layers
        params, grads = [], []
        for i in range(num_layers):
            params += layers[i].params
            grads += layers[i].gradients
        for i in range(len(grads)):
            params[i] = params[i] - learning_rate*grads[i]
            
        ####################################################
        #            END OF YOUR CODE                      #
        ####################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        
    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class          #
        # SoftmaxLayer                                      #
        #####################################################
        ####################################################
        #           START OF YOUR CODE                     #
        ####################################################
        
        for layer in layers:
            X = layer.feedforward(X)
        
        softmax = np.exp(X - np.max(X, axis=1, keepdims=True))
        softmax /= np.sum(softmax, axis=1, keepdims=True)
        
        predictions = np.argmax(softmax, axis=1)

        ####################################################
        #            END OF YOUR CODE                      #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


