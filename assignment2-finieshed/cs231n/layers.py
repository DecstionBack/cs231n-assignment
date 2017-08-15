#encoding=utf-8
import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #x: N*D  w:D*M b:M*1  M:number of classes
  #x = x.reshape(x.shape[0], -1) 
  out = np.dot(x.reshape(x.shape[0], -1),w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)  dout * 1
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #
#w:50*7 x:3*50 dout:number
#Determine the dot order and the sum of b(axis=0/1) according to the shape of x,w,b,dout
  dx =  np.dot(dout, w.T).reshape(x.shape)
  dw=np.dot(x.reshape(x.shape[0], -1).T, dout)
  db=(np.sum(dout, axis=0)).T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################

  out = np.maximum(0,x)
   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################

  dx = dout
  dx[x<0] = 0

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  该函数的目的是把传入的w*x+b得到的数据正规化并且保留一部分数据供以后使用。
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.
  这儿是什么意思？计算均值和方差采用mini-batch选取的样本，然后记录下来别的变量用于在测试的时候使用？？怎么使用的？？

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    
    sample_mean = np.mean(x, axis=0)
    
    sample_var = np.std(x, axis=0)**2
    
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    
    out = gamma * x_hat + beta
    #cache不能同时保存x和x_hat，只需要保留参数，然后变换即可得到x_hat
    cache = (x, gamma, beta, sample_mean, sample_var, eps)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    #pass
    sample_mean = np.mean(x, axis=0)
    sample_var = np.std(x, axis=0)**2
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    x_t = (x - sample_mean) / np.sqrt(sample_var)
    out = gamma * x_t + beta
    cache = ()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
#affine_forward-->batch_normalization-->Relu-->
  dx, dgamma, dbeta = None, None, None
  (x, gamma, beta, sample_mean, sample_var, eps) = cache
  num = dout.shape[0]
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  #参见论文中对beta的求导结果，可以分开来看，每次对beta_i求导，每次会得到一个行向量， 随后如果是对整个beta求导，则是这些行向量加在一起，也就是下面的公式
  dbeta = np.sum(dout, axis=0) 
    
  
  x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
  dgamma = np.sum(dout * x_hat, axis = 0)
  
  #下面的推导过程整理出来公式保存起来，以后忘了的时候再重新回来看
  #需要补充：求导链式法则；矩阵、向量之间求导的维度问题
  d_x_hat = dout * gamma 
  
  avg = np.mean(x,axis=0)
  var = np.std(x, axis=0) ** 2
  d_var = np.sum(d_x_hat * (x - avg) * (-0.5) * (var + eps) **(-1.5), axis=0)
  d_avg = np.sum(d_x_hat,axis=0) * (-1)/ np.sqrt(var + eps) + d_var * np.sum((-2.0/num) * (x - avg),axis=0)
  dx = d_x_hat * (1.0 / np.sqrt(var + eps)) + d_var * 2.0 / num * (x - avg) + d_avg / num

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  (x, gamma, beta, sample_mean, sample_var, eps) = cache
  num = dout.shape[0]
  dbeta = np.sum(dout, axis=0) 
  x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
  dgamma = np.sum(dout * x_hat, axis = 0)

  avg = np.mean(x,axis=0)
  var = np.std(x, axis=0) ** 2
  d_x_hat = dout * gamma 
  
  dx=(d_x_hat - np.mean(d_x_hat,axis=0))/np.sqrt(var+eps) - (np.mean(d_x_hat* (x-avg),axis=0))*(x-avg)/(var+eps)**1.5
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p ) / p
    out = x * mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    mask = np.ones_like(x)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    dx = dout * mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W) N个数 H高度 W宽度 C深度
  - w: Filter weights of shape (F, C, HH, WW)  F-filter个数 HH高度 WW宽度 C深度，与X的一致
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  #out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.  np.pad具体含义？？                      #
  #############################################################################
  
  N, C, H ,W =x.shape
  F, C, HH, WW = w.shape

  stride = conv_param['stride']
  pad = conv_param['pad']
  #在原始的x两边打上pad，x中的N和C与pad无关，只有后面的H和W才需要两边都加上一层0
  x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)) , 'constant', constant_values=0)
  #print x.shape
  #print x_pad.shape
  #x.shape=[2, 3, 4, 4], x_pad.shape=[2, 3, 6, 6],加上pad以后的效果
  H_new = 1 + (H + 2 * pad -HH) / stride
  W_new = 1 + (W + 2 * pad -WW) / stride
  out = np.zeros([N, F, H_new, W_new])
  #i和f比较好判断，x_pad的下标可以自己画一个示意图简单推导一下
  #N在输入和输出都是一样的，只需要考虑F个filter对一个输入数据得到的结果即可，对后面的N-1个输入的处理是一样的
  #N是第几个输入，f是深度层，j和k代表二维平面中的卷积结果
  for i in range(0,N):
        for f in range(0,F):
            for j in range(0,H_new):
                  for k in range(0,W_new):
                    #out[i,j,k,t]应该是一个数，这里t换成了:，应该是一个列向量  这里+=可以写可以不写，因为本身out对应的位置为0，+=和=效果一样，
                    #而且这里+=没有明确的物理意义，每一个out中的元素只是计算了一次
                    out[i,f,j,k] += np.sum(x_pad[i,:, stride*j:stride*j+HH,stride*k:stride*k+WW]* w[f,:,:,:]) + b[f]
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  x, w, b, conv_param = cache
  #db：每一个out中的元素只与一个b有关，所以只需要讲其余维度的卷起来求和得到一个F维度的向量即可
  db = np.sum(dout, axis=(0,2,3))
  
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)

  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  
  H_new = (H + 2*pad - HH)/stride +1
  W_new = (W + 2*pad - WW)/stride +1
    
  #x的维度为N-D-H-W，只有H和W两个维度需要加上两边的pad
  x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant')
  dx_pad = np.pad(dx, ((0,),(0,),(pad,),(pad,)), 'constant')

  for i in range(0,N):
        for f in range(0,F):
            for j in range(0,H_new):
                  for k in range(0,W_new):
                    #out[i,f,j,k] = np.sum(x_pad[i,:, stride*j:stride*j+HH,stride*k:stride*k+WW]* w[f,:,:,:]) + b[f]
                    #按照计算的时候反向求导，不用太去形象化理解
                    #每一个out都是按照上面的式子计算的，这里按照原来的循环规则一次一次把计算出来的对应的out对w和x的导数求出来然后多个矩阵相加即可，不要深入进去一个一个地想细节
                    #前面的out计算公式可以看成是A+=np.sum(B*w[f,:,:,:])，则对w[f,:,:,:]求导就是dA * (dA/dw)，则为dout[i,f,j,k]*B,代入即可，对dx_pad的求导方式是一样的。
                    dw[f,:,:,:] += x_pad[i,:, stride*j:stride*j+HH, stride*k:stride*k+HH] * dout[i,f,j, k]
            
                    dx_pad[i,:,stride*j:stride*j+WW, stride*k:stride*k+WW] += w[f,:,:,:] * dout[i,f,j, k]
  #因为计算out的时候使用的是x_pad，所以这里需要写dx_pad，为了和dx区分
  #dx只是计算dx_pad中的出去两边的pad的部分，所以从dx_pad中取出来
  dx = dx_pad[:,:, pad:H+pad, pad:W+pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  
  #返回的结果的宽度和高度
  width = (W - pool_width) / stride + 1
  height = (H - pool_height) / stride + 1
    
  out = np.zeros((N,C,height, width))
    
  for i in range(0,N):
        for c in range(0,C): #这里的C实际上是Filter的个数F，卷积得到的结果Filter的个数为结果的深度
            for j in range(0,height):
                  for k in range(0,width):
                        #这里的x是经过卷积以后的x，所以NCHW,在计算的时候，i和c保持不变，只是H和W缩小到height和width了
                        out[i,c,j,k] = np.max(x[i,c,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width])
  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
    
  width = (W - pool_width) / stride + 1
  height = (H - pool_height) / stride + 1
    
  dx = np.zeros_like(x)


  for i in range(0,N):
        for c in range(0,C): #这里的C实际上是Filter的个数F，卷积得到的结果Filter的个数为结果的深度
            for j in range(0,height):
                  for k in range(0,width):
                      # out[i,c,j,k] = np.max(x[i,c,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width])
                      #max对应的位置为1，而其余的位置为0
                      #这里写chooseArea是因为如果不单独写一个变量，后面会重复写两遍，式子太长了
                      chooseArea = x[i,c,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width]
                      dx[i,c,stride*j:stride*j+pool_height,stride*k:stride*k+pool_width] =  (chooseArea==np.max(chooseArea))   * dout[i,c,j,k]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  out = np.zeros_like(x)
  cache = {}
  N,C,H,W = x.shape
#对每一个通道做批量归一化处理
  for c in range(0,C):
    x_re = x[:,c,:,:].reshape((N,-1))
    out_bn, cache_bn = batchnorm_forward(x_re, gamma[c], beta[c], bn_param)
    out[:,c,:,:] = out_bn.reshape(N,H,W)
    cache['cache_bn'+str(c)] = cache_bn
    
  cache['x'] = x
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  x = cache['x']
  N,C,H,W = x.shape

  dx = np.zeros_like(x)
  dgamma = np.zeros(C)
  dbeta = np.zeros(C)
  
  for c in xrange(C):
      #x_re = x[:,c,:,:].reshape((N,-1))，后面传过来的梯度dout_re与dout[:,c,:,:]相同
      dout_re = dout[:,c,:,:].reshape((N,-1))
      dx_bn, dgamma_bn, dbeta_bn = batchnorm_backward_alt(dout_re, cache['cache_bn'+str(c)])
      #前面都是把x展开去运算，这里要恢复原来的维度
      dx[:,c,:,:] = dx_bn.reshape(N,H,W)
      #上面的backward传入的gamma和beta都只是gamma和beta向量中的一个变量，所以求导之后要全部加和起来
      dgamma[c], dbeta[c] = np.sum(dgamma_bn), np.sum(dbeta_bn)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
