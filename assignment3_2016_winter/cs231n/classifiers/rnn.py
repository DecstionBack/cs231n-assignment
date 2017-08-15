#encoding=utf-8
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
  """
  A CaptioningRNN produces captions from image features using a recurrent
  neural network.

  The RNN receives input vectors of size D, has a vocab size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses word vectors
  of dimension W, and operates on minibatches of size N.
  D：输入的向量维度     V：词汇表的长度，D小于等于V的   T：时间步的总数   H：隐藏层的参数维度？     W: 隐藏层的词向量维度    N：一次选取的N个样本

  Note that we don't use any regularization for the CaptioningRNN.  不需要正则化来防止过拟合
  """
  
  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32):
    """
    Construct a new CaptioningRNN instance.

    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
      and maps each string to a unique integer in the range [0, V).  这里简单起见，将每一个词语都对应到一个整数而已
    - input_dim: Dimension D of input image feature vectors.   输入的图像特征向量的维度D，input输入的是图像特征
    - wordvec_dim: Dimension W of word vectors.  词向量的特征数W，Image Captioning中需要图像和词语，这是两大部分
    - hidden_dim: Dimension H for the hidden state of the RNN.  RNN隐藏层的维度H？？
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'. 决定是RNN还是LSTM
    - dtype: numpy datatype to use; use float32 for training and float64 for  检测梯度时使用float64更精确，训练时使用float32更省空间
      numeric gradient checking.
    """
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}   #self.idx_to_word是dict类型，将word_to_idx的key和value颠倒了
    self.params = {}
    
    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    
    
    #这儿的初始化需要好好理解一下！！
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)   #总共有vocab_size个词语，每个词语对应的向量是wordvec_dim维
    self.params['W_embed'] /= 100    #这儿除以100是什么意思？？？
    
    # Initialize CNN -> hidden state projection parameters
    #这儿的W_proj是不是可以看成是课件中的Wih??
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)   #input_dim输入的是图像特征，而不是文本
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)  #Wh相当于从一个隐藏层到另外一个隐藏层，所以是hidden_dim*hidden_dim
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)  #输出也是向量，每一次输出一个1*vocab_size的向量，每一个数字代表对应词语的结果，选取最大的作为输出。这里的vocab_size指的应该是此表的尺寸，也就是要输出后面输出每一个词语的概率，然后选择最大的那个，所以维度是hidden_dim * vocab_size
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      
    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, captions):
    """
    Compute training-time loss for the RNN. We input image features and
    ground-truth captions for those images, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.
    
    Inputs:
    - features: Input image features, of shape (N, D)   N是minibatch的样本大小，D是一个图像被拉成的向量维度
    - captions: Ground-truth captions; an integer array of shape (N, T) where  N和上面同样，T是caption的单词数目吧，一次输入一个单词
      each element is in the range 0 <= y[i, t] < V
      
    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut captions into two pieces: captions_in has everything but the last word
    # and will be input to the RNN; captions_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of captions_in will be the START
    # token, and the first element of captions_out will be the first word.
    captions_in = captions[:, :-1]  #caption_in只是取captions的前1到T-1，而caption_out只是去captions的2到T
    captions_out = captions[:, 1:]
    
    # You'll need this 
    mask = (captions_out != self._null)  #self._null=word_to_idx['<NULL>']，也是数字，代表<NULL>所对应的数字

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    #将CNN得到的向量拉伸以后输入RNN，这里的W_proj是将其进行变换额外设置的，应该是课件中的Wih。
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
    # In the forward pass you will need to do the following:                   #
    # (1) Use an affine transformation to compute the initial hidden state     #
    #     from the image features. This should produce an array of shape (N, H)#
    # (2) Use a word embedding layer to transform the words in captions_in     #
    #     from indices to vectors, giving an array of shape (N, T, W).         #
    # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
    #     process the sequence of input word vectors and produce hidden state  #
    #     vectors for all timesteps, producing an array of shape (N, T, H).    #
    # (4) Use a (temporal) affine transformation to compute scores over the    #
    #     vocabulary at every timestep using the hidden states, giving an      #
    #     array of shape (N, T, V).                                            #
    # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
    #     the points where the output word is <NULL> using the mask above.     #
    #                                                                          #
    # In the backward pass you will need to compute the gradient of the loss   #
    # with respect to all model parameters. Use the loss and grads variables   #
    # defined above to store loss and gradients; grads[k] should give the      #
    # gradients for self.params[k].                                            #
    ############################################################################
    #D-input_dim  H:hidden_dim 
    image_features = np.dot(features, W_proj) + b_proj  #N*H
    word_vectors, word_cache = word_embedding_forward(captions_in, W_embed)  #word_vectors:N*T*W
    #h0默认初始化为全0的
    #h0 = np.zeros((N, H))
    #按理说应该传入一个全都是0的h0，但是由于接口中没有让传入h0这一个环节，所以这里可以将h0换为image_features，因为根据公式后面需要加上这一项
    h0 = image_features
    if self.cell_type == 'rnn':
        h, h_cache = rnn_forward(word_vectors, h0, Wx, Wh, b)  #h:N*T*H
    elif self.cell_type == 'lstm':
        h,h_cache = lstm_forward(word_vectors, h0, Wx, Wh, b)
    
    scores,score_cache = temporal_affine_forward(h, W_vocab, b_vocab)  #sccores:N*T*V  W_vocab应该是课件中的Why,即计算输出结果需要的参数。在前向网络中，输出层的神经元个数可以看做是每个分类所得到的分数，而这儿每一个时间步的输出y可以看做是一个神经元的输出，通过y就可以计算损失函数。在时间步增加的过程中，每一次输出的结果对一个样本而言实际上是一个和标准答案相同维度的矩阵，如果是词语，则是一个列向量，多个时间步以后可以将这多个列向量拼合成一个矩阵，所以在计算总体损失的时候可以直接计算矩阵和标准答案组成的矩阵之间的损失即可，而不用每一个时间步都计算一遍再累加。
    
    loss, dscores = temporal_softmax_loss(scores, captions_out, mask)
    #训练过程可以这样理解：T是时间步，每一次的输入维度为N*W，N是传入的样本的数量，W是每一个词语的向量维度，这样子在每一个时间步，每一句话都去第i个词语的向量当做输入，与隐藏状态h进行作用之后得到新的状态，一直持续下去，直到第T步，算作一轮训练。由于在训练时要求词语的数目是一致的，所以在一些短的语句后面计入<NULL>来占位。与此同时，每一次的输入captions_in的标准答案存储在你captions_out中，这样在输入一次得到结果输出时，可以计算当前时间步i的误差Li。
    
    dx, dW_vocab, db_vocab=temporal_affine_backward(dscores, score_cache)
    if self.cell_type == 'rnn':
        dx, dh0, dWx, dWh, db = rnn_backward(dx, h_cache)
    elif self.cell_type == 'lstm':
        dx, dh0, dWx, dWh, db = lstm_backward(dx, h_cache)
    dW_embed = word_embedding_backward(dx, word_cache)
    
    grads['W_vocab'],grads['b_vocab'],grads['Wx'],grads['Wh'],grads['b'],grads['W_embed'] = dW_vocab,db_vocab,dWx,dWh,db,dW_embed
    
    #得到dh0以后应该对dh进行更新，但是这里的image_features是输入的向量，不能变，所以需要对随机生成的矩阵W_proj来进行更新
    dW_proj = np.dot(features.T, dh0)
    db_proj = np.sum(dh0, axis=0)
    grads['W_proj'], grads['b_proj'] = dW_proj, db_proj

    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


  def sample(self, features, max_length=30):
    """
    Run a test-time forward pass for the model, sampling captions for input
    feature vectors.

    At each timestep, we embed the current word, pass it and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    scores for all vocab words, and choose the word with the highest score as
    the next word. The initial hidden state is computed by applying an affine 初始的隐藏状态h0=image_features，输入第一个caption是<START>
    transform to the input image features, and the initial word is the <START>
    token.

    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.

    Inputs:
    - features: Array of input image features of shape (N, D).
    - max_length: Maximum length T of generated captions.

    Returns:
    - captions: Array of shape (N, max_length) giving sampled captions,
      where each element is an integer in the range [0, V). The first element
      of captions should be the first sampled word, not the <START> token.
    """
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)   #有N个图片特征输入，应该是N个caption序列输出，初始时先都设为<NULL>，长度都为max_length，如果长度不够，则相当于用<NULL>来填充了

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    ###########################################################################
    # TODO: Implement test-time sampling for the model. You will need to      #
    # initialize the hidden state of the RNN by applying the learned affine   #
    # transform to the input image features. The first word that you feed to  #
    # the RNN should be the <START> token; its value is stored in the         #
    # variable self._start. At each timestep you will need to do to:          #
    # (1) Embed the previous word using the learned word embeddings           #
    # (2) Make an RNN step using the previous hidden state and the embedded   #
    #     current word to get the next hidden state.                          #
    # (3) Apply the learned affine transformation to the next hidden state to #
    #     get scores for all words in the vocabulary                          #
    # (4) Select the word with the highest score as the next word, writing it #
    #     to the appropriate slot in the captions variable                    #
    #                                                                         #
    # For simplicity, you do not need to stop generating after an <END> token #
    # is sampled, but you can if you want to.                                 #
    #                                                                         #
    # HINT: You will not be able to use the rnn_forward or lstm_forward       #
    # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
    # a loop.                                                                 #
    ###########################################################################
    
    word_id = [self._start] * N    #输入的单词其实是输入下标的，不需要先构造出向量，在循环中W_embed会得到对应的向量
    T = max_length
    prev_h = np.dot(features, W_proj) + b_proj
    next_c = np.zeros_like(prev_h)
   
    for i in xrange(T):
        word_vector = W_embed[word_id]
        if self.cell_type == 'rnn':
            next_h, h_cache = rnn_step_forward(word_vector, prev_h, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            next_h,next_c, h_cache = lstm_step_forward(word_vector, prev_h, next_c, Wx, Wh, b)
        prev_h = next_h
        scores, scores_cache = temporal_affine_forward(next_h.reshape(N,1,next_h.shape[1]),W_vocab,b_vocab)
        
        scores = scores.reshape(scores.shape[0],scores.shape[2])
        new_word_id =  np.argmax(scores, axis=1)   #去除每一个样本得到的分数中结果最大的下表
        word_id = new_word_id  #选出得分较高的单词作为下一次的输入
        #word_id代表每一个样本输出的单词，维度为N*1

        for j in xrange(N):
            captions[j,i] = new_word_id[j]
             
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return captions
