#encoding=utf-8
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from math import sqrt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    W1 = np.random.randn(input_dim, hidden_dim)*weight_scale
    W2 = np.random.randn(hidden_dim, num_classes)*weight_scale
    b1=np.zeros(hidden_dim)
    b2=np.zeros(num_classes)
    #b1 and b2 are both column vectors
    #print b1.shape
    #print b2.shape
    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['b1'] = b1
    self.params['b2'] = b2
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']
    a1 , cache_hidden_layer = affine_relu_forward(X, W1, b1)
    scores, cache_scores = affine_forward(a1, W2, b2)
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    #softmax_loss will compute the dataloss and dx
    #x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    #for the ith input. Not our original data
    dataloss, dscores = softmax_loss(scores,y)
    loss = dataloss + 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
     
    da2, dW2, db2 = affine_backward(dscores, cache_scores)   #dscores, not np.ones
    dx, dW1, db1 = affine_relu_backward(da2, cache_hidden_layer)
    #这里不用让dW1和dW2再除以x.shape[0]，因为在上面调用的softmax_loss中已经做了这一步，求解的是第一部分，后面只需要加上正则项就可以了
    grads['W2'] = dW2 + self.reg * W2
    grads['W1'] = dW1 + self.reg * W1
    grads['b2'] = db2
    grads['b1'] = db1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    #pass
    #If we have n-hidden layer, then we have (n+1)<W,b> pairs, W1 and Wn+1 is dealt with seperately
    W1 = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
    b1 = np.zeros(hidden_dims[0])
    self.params['W1'] = W1
    self.params['b1'] = b1
    hidden_layer = len(hidden_dims)
    for i in xrange(hidden_layer - 1):
        W = np.random.randn(hidden_dims[i], hidden_dims[i+1]) * weight_scale
        b = np.zeros(hidden_dims[i+1])
        self.params['W' + str(i+2)] = W
        self.params['b' + str(i+2)] = b
    Wn = np.random.randn(hidden_dims[-1], num_classes) * weight_scale
    bn = np.zeros(num_classes)
    self.params['W' + str(hidden_layer + 1)] = Wn
    self.params['b' + str(hidden_layer + 1)] = bn
        
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    #pass
    #hidden_layer = len(self.hidden_dims)
    #类中没有hidden_dims变量，但是却有num_layers变量，num_layers = hidden_layer + 1，没有区别
    hidden_layer = self.num_layers - 1
    cache_history = []
    #w和b在计算的时候中间变量没有特别说明是下标，因为只是作为中间变量出现而已，但是每一次的cache都需要保存，在反向求导的时候会用到
    X = X.reshape(X.shape[0],-1)
    for i in xrange(hidden_layer):
        w = self.params['W' + str(i+1)]
        b = self.params['b' + str(i+1)]
        a_i, cache_i = affine_relu_forward(X, w, b)
        cache_history.append(cache_i)
        X = a_i
    Wn = self.params['W' + str(hidden_layer + 1)]
    bn = self.params['b' + str(hidden_layer + 1)]
    scores, score_cache = affine_forward(X, Wn, bn)
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #pass
    #y在函数调用的时候会传过来，如果不传入y则只是test，不需要计算loss和grads
    loss, dscores = softmax_loss(scores, y)
    #W1...W(hidden_layer+1)
    for i in xrange(1, hidden_layer + 2):
        loss += 0.5 * self.reg * np.sum((self.params['W' + str(i)] ** 2))
    
    da_final, dw_final, db_final = affine_backward(dscores, score_cache)
    grads['W' + str(hidden_layer + 1)] = dw_final + self.reg * self.params['W' + str(hidden_layer + 1)]
    grads['b' + str(hidden_layer + 1)] = db_final
    dout = da_final
    #cache_history不包括最后一层的cache，所以长度和hidden_layer大小一样
    for i in range(hidden_layer, 0, -1):
        da, dw, db = affine_relu_backward(dout, cache_history[i-1])
        grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
        grads['b' + str(i)] = db
        dout = da
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
