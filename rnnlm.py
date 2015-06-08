from numpy import *
import itertools
import time
import sys
from misc import random_weight_matrix

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid,make_onehot
from nn.math import MultinomialSampler, multinomial_sample


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)] + b1)
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        #random.seed(rseed)
        self.params.U=0.1*random.randn(*self.params.U.shape)
        self.sparams.L=0.1*random.randn(*self.sparams.L.shape)
        self.params.H=random_weight_matrix(*self.params.H.shape)
        self.bptt=bptt
        self.alpha=alpha
        # Initialize word vectors

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H)
                and self.sgrads (for L,U)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        for i in range(ns):
            hs[i]=sigmoid(self.params.H.dot(hs[i-1])+self.sparams.L[xs[i]])
            ps[i]=softmax(self.params.U.dot(hs[i]))

        ##
        # Backward propagation through time
        #for i in range(ns-1,-1,-1):
            y = make_onehot(ys[i], self.vdim)
            delta1 = ps[i]-y
            self.grads.U += outer(delta1, hs[i])
            for j in range(self.bptt):
                t = i - j
                if t < 0:
                    break
                if j == 0:
                    delta = (hs[t]*(1-hs[t]))*(self.params.U.T.dot(delta1))
                else:
                    delta = (hs[t]*(1-hs[t]))*(self.params.H.T.dot(delta))
                self.grads.H += outer(delta,hs[t-1])
                self.sgrads.L[xs[t]] = delta


        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        ns = len(ys)
        hs = zeros((ns+1, self.hdim))
        ps = zeros((ns, self.vdim))
        for i in range(ns):
            hs[i]=sigmoid(self.params.H.dot(hs[i-1])+self.sparams.L[xs[i]])
            ps[i]=softmax(self.params.U.dot(hs[i]))
            J-=log(ps[i][ys[i]])
        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """
        J = 0 # total loss
        ys = [init] # emitted sequence


        #### YOUR CODE HERE ####
        hs = zeros((maxlen + 1, self.hdim))
        ps = zeros((maxlen, self.vdim))
        t = 0
        while ys[-1] != end and t < maxlen: 
            # print ys[-1]
            hs[t, :] = sigmoid((self.params.H.dot(hs[t - 1, :].T)).T + self.sparams.L[ys[-1], :])
            ps[t, :] = softmax(self.params.U.dot(hs[t, :].T)).T
            # y = argmax(ps[t, :])
            y = multinomial_sample(ps[t, :])
            ys.append(y)
            J += - log(ps[t, y])
            t += 1
        #### YOUR CODE HERE ####
        return ys, J

    # def mysoftmax(x):
    #     maxs=max(x)
    #     for i in range(len(x)):
    #         x[i]-=maxs[i]
    #     x=exp(x)
    #     sums=sum(x,axis=1)
    #     for i in range(a):
    #         x[i]/=sums[i]
    
    #     return x



class RNNPT(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)] + b1)
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        #random.seed(rseed)
        self.params.U=0.1*random.randn(*self.params.U.shape)
        self.sparams.L=0.1*random.randn(*self.sparams.L.shape)
        self.params.H=random_weight_matrix(*self.params.H.shape)
        self.bptt=bptt
        self.alpha=alpha
        # Initialize word vectors

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys, h0):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H)
                and self.sgrads (for L,U)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        hs[-1] = h0
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        for i in range(ns):
            hs[i]=sigmoid(self.params.H.dot(hs[i-1])+self.sparams.L[xs[i]])
            ps[i]=softmax(self.params.U.dot(hs[i]))

        ##
        # Backward propagation through time
        #for i in range(ns-1,-1,-1):
            y = make_onehot(ys[i], self.vdim)
            delta1 = ps[i]-y
            self.grads.U += outer(delta1, hs[i])
            for j in range(self.bptt):
                t = i - j
                if t < 0:
                    break
                if j == 0:
                    delta = (hs[t]*(1-hs[t]))*(self.params.U.T.dot(delta1))
                else:
                    delta = (hs[t]*(1-hs[t]))*(self.params.H.T.dot(delta))
                self.grads.H += outer(delta,hs[t-1])
                self.sgrads.L[xs[t]] = delta


        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys, h0):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        ns = len(ys)
        hs = zeros((ns+1, self.hdim))
        hs[-1] = h0
        ps = zeros((ns, self.vdim))
        for i in range(ns):
            hs[i]=sigmoid(self.params.H.dot(hs[i-1])+self.sparams.L[xs[i]])
            ps[i]=softmax(self.params.U.dot(hs[i]))
            J-=log(ps[i][ys[i]])
        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y, H0):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y, H0)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys, h0)
                       for xs,ys, h0 in itertools.izip(X, Y, H0)])

    def compute_mean_loss(self, X, Y, H0):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y, H0)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, h0, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """
        J = 0 # total loss
        ys = [init] # emitted sequence


        hs = zeros((maxlen + 1, self.hdim))
	hs[-1] = h0
        ps = zeros((maxlen, self.vdim))
        t = 0
        while ys[-1] != end and t < maxlen: 
            # print ys[-1]
            hs[t, :] = sigmoid((self.params.H.dot(hs[t - 1, :].T)).T + self.sparams.L[ys[-1], :])
            ps[t, :] = softmax(self.params.U.dot(hs[t, :].T)).T
            # y = argmax(ps[t, :])
            y = multinomial_sample(ps[t, :])
            ys.append(y)
            J += - log(ps[t, y])
            t += 1
        return ys, J

