import sys, os
from numpy import *
from matplotlib.pyplot import *

matplotlib.rcParams['savefig.dpi'] = 100

from rnnlm import RNNLM, RNNPT
from data_utils import utils as du
import pandas as pd
import itertools

def adjust_loss(loss, funk):
    return (loss + funk * log(funk))/(1 - funk)


def epochiter(N, nepoch=5):
    
       # """Iterator to loop sequentially through training sets."""
    return itertools.chain.from_iterable(itertools.repeat(xrange(N), nepoch))


# Load the vocabulary
vocab = pd.read_table("data/dictionary", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )
# Choose how many top words to keep
vocabsize = len(vocab)
print 'vocabulary size %d' % vocabsize
#vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)
print 'load dictionary done'
docs = du.load_dataset('data/rnn_input')
S_train = du.docs_to_indices(docs, word_to_num)
X_train, Y_train = du.seqs_to_lmXY(S_train)
#X_train = X_train[:3000]
#Y_train = Y_train[:3000]
print 'load data done'
print 'number of training data %d' % len(Y_train)

# Load the dev set (for tuning hyperparameters)
#docs = du.load_dataset('data/lm/ptb-dev.txt')
#S_dev = du.docs_to_indices(docs, word_to_num)
#X_dev, Y_dev = du.seqs_to_lmXY(S_dev)

# Load the test set (final evaluation only)
#docs = du.load_dataset('data/lm/ptb-test.txt')
#S_test = du.docs_to_indices(docs, word_to_num)
#X_test, Y_test = du.seqs_to_lmXY(S_test)
model = "RNNLM"

if model == "RNNLM"
	hdim = 40 # dimension of hidden layer = dimension of word vectors
	#random.seed(10)
	L0 = zeros((vocabsize, hdim)) # replace with random init, 
	                              # or do in RNNLM.__init__()
	model = RNNLM(L0, U0 = L0, alpha=0.1,  bptt=3)

	nepoch = 1
	N = nepoch * len(Y_train)
	k = 5 # minibatch size
	#random.seed(10)
	#idx=[]
	#print X_train.size
	#for i in range(N/k):
	#    idx.append(random.choice(len(Y_train),k))
	idx = epochiter(len(Y_train), nepoch)
	model.train_sgd(X = X_train, y = Y_train, idxiter = idx, printevery = 500, costevery = 500)

	#dev_loss = model.compute_mean_loss(X_dev, Y_dev)
	if not os.path.exists("model/" + model):
		os.makedirs("model/" + model)

	#print "Unadjusted: %.03f" % exp(dev_loss)
	#print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
	save("model/" + model + "/rnnlm.L.npy", model.sparams.L)
	save("model/" + model + "/rnnlm.U.npy", model.params.U)
	save("model/" + model + "/rnnlm.H.npy", model.params.H)


elif model == "RNNPT":
	hdim = 40 # dimension of hidden layer = dimension of word vectors
	#random.seed(10)
	L0 = zeros((vocabsize, hdim)) # replace with random init, 
	                              # or do in RNNLM.__init__()
	model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

	nepoch = 1
	N = nepoch * len(Y_train)
	k = 5 # minibatch size
	#random.seed(10)
	#idx=[]
	#print X_train.size
	#for i in range(N/k):
	#    idx.append(random.choice(len(Y_train),k))
	h0_train = pickle.load(open('data/h0_train', 'rb'))
	idx = epochiter(len(Y_train), nepoch)
	model.train_sgd_rnnpt(X = X_train, y = Y_train, h0 = h0_train, idxiter = idx, printevery = 500, costevery = 500)

	#dev_loss = model.compute_mean_loss(X_dev, Y_dev)
	if not os.path.exists("model/" + model):
		os.makedirs("model/" + model)

	#print "Unadjusted: %.03f" % exp(dev_loss)
	#print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
	save("model/" + model + "/rnnlm.L.npy", model.sparams.L)
	save("model/" + model + "/rnnlm.U.npy", model.params.U)
	save("model/" + model + "/rnnlm.H.npy", model.params.H)


elif model == "RNNPTONE":
	hdim = 40 # dimension of hidden layer = dimension of word vectors
	#random.seed(10)
	L0 = zeros((vocabsize, hdim)) # replace with random init, 
	                              # or do in RNNLM.__init__()
	model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

	nepoch = 1
	N = nepoch * len(Y_train)
	k = 5 # minibatch size
	#random.seed(10)
	#idx=[]
	#print X_train.size
	#for i in range(N/k):
	#    idx.append(random.choice(len(Y_train),k))
	h0_train = pickle.load(open('data/h0_train', 'rb'))
	idx = epochiter(len(Y_train), nepoch)
	model.train_sgd_rnnpt(X = X_train, y = Y_train, h0 = h0_train, idxiter = idx, printevery = 500, costevery = 500)

	#dev_loss = model.compute_mean_loss(X_dev, Y_dev)
	if not os.path.exists("model/" + model):
		os.makedirs("model/" + model)

	#print "Unadjusted: %.03f" % exp(dev_loss)
	#print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
	save("model/" + model + "/rnnlm.L.npy", model.sparams.L)
	save("model/" + model + "/rnnlm.U.npy", model.params.U)
	save("model/" + model + "/rnnlm.H.npy", model.params.H)