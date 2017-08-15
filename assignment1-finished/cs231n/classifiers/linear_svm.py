import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #W:3073*10, X:49000*3073, y:49000*1
  # compute the loss and the gradient
  num_classes = W.shape[1]    #num_classes = 10
  num_train = X.shape[0]  #num_train = 49000
  loss = 0.0
  for i in xrange(num_train): 
    scores = X[i].dot(W)   #XiW
    correct_class_score = scores[y[i]]   #Scores of correct class, for an input it is a constant
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #Compute dW, dW will change for every X[i]
        dW[:,j] += X[i,:].T   #sum all of the X[i] don't forget the sum symbol
        dW[:,y[i]] -= X[i,:].T
        
  # Right now the loss is a sum over all 
  
  loss = loss / num_train
  loss += 0.5* reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  scores = X.dot(W)
  num_train = X.shape[0]
  row_index = np.arange(num_train)
  margin = scores - scores[row_index, y].reshape([num_train,1]) + 1  # every score will substract f(w,x)yi
  margin[row_index,y] = 0
  loss = np.sum(margin[margin>0]) / num_train
  loss+= 0.5 * reg * np.sum(W * W)
  
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  L = np.zeros(margin.shape)
  L[margin>0]=1
  L[np.arange(num_train),y] -= np.sum(margin>0, axis=1)  #margin??why??
  dW = np.dot(X.T, L) / num_train
  dW += reg*W   #margin >0 or not ,we will add reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
