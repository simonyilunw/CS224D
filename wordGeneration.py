import sys, os
from numpy import *
from matplotlib.pyplot import *

matplotlib.rcParams['savefig.dpi'] = 100

from rnnlm import RNNLM, RNNPT
from data_utils import utils as du
import pandas as pd
import itertools
import cPickle as pickle

def adjust_loss(loss, funk):
	return (loss + funk * log(funk))/(1 - funk)


def epochiter(N, nepoch=5):
    
       # """Iterator to loop sequentially through training sets."""	
       return itertools.chain.from_iterable(itertools.repeat(xrange(N), nepoch))

def toONE(h0):
	new = []
	for h in h0:
		newh = zeros_like(h)
		newh[:5] = h[:5]
		tmp = argmax(h[5:]) + 5
		newh[tmp] = 1
		new.append(newh)
	return new


if __name__ == "__main__":

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
	docs = du.load_dataset('data/rnn_input_train')
	S_train = du.docs_to_indices(docs, word_to_num)
	X_train, Y_train = du.seqs_to_lmXY(S_train)

	docs = du.load_dataset('data/rnn_input_test')
	S_train = du.docs_to_indices(docs, word_to_num)
	X_dev, Y_dev = du.seqs_to_lmXY(S_train)
	#X_train = X_train[:3000]
	#Y_train = Y_train[:3000]
	print 'load data done'
	print 'number of training data %d' % len(Y_train)

	method = "RNNPTONE"
	hdim = 40 # dimension of hidden layer = dimension of word vectors
	#random.seed(10)
	nepoch = 1
	N = nepoch * len(Y_train)
	k = 5 # minibatch size
	fraction_lost = 0.07923163705
	#idx=[]
	#print X_train.size
	#for i in range(N/k):
	#    idx.append(random.choice(len(Y_train),k))
	if method == "RNNLM":
		L0 = zeros((vocabsize, hdim)) # replace with random init, 
					      # or do in RNNLM.__init__()
		model = RNNLM(L0, U0 = L0, alpha=0.1,  bptt=3)

		idx = epochiter(len(Y_train), nepoch)
		model.train_sgd(X = X_train, y = Y_train, idxiter = idx, printevery = 500, costevery = 500)

		dev_loss = model.compute_mean_loss(X_dev, Y_dev)
		if not os.path.exists("model/" + method):
			os.makedirs("model/" + method)

		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		save("model/" + method + "/rnnlm.L.npy", model.sparams.L)
		save("model/" + method + "/rnnlm.U.npy", model.params.U)
		save("model/" + method + "/rnnlm.H.npy", model.params.H)


	elif method == "RNNPT":
		#random.seed(10)
		L0 = zeros((vocabsize, hdim)) # replace with random init, 
					      # or do in RNNLM.__init__()
		model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

		h0_train = pickle.load(open('data/h0_train', 'rb'))
		h0_test = pickle.load(open('data/h0_test', 'rb'))
		idx = epochiter(len(Y_train), nepoch)
		model.train_sgd_rnnpt(X = X_train, y = Y_train, h0 = h0_train, idxiter = idx, printevery = 500, costevery = 500)

		dev_loss = model.compute_mean_loss(X_dev, Y_dev, h0_test)
		if not os.path.exists("model/" + method):
			os.makedirs("model/" + method)

		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		save("model/" + method + "/rnnlm.L.npy", model.sparams.L)
		save("model/" + method + "/rnnlm.U.npy", model.params.U)
		save("model/" + method + "/rnnlm.H.npy", model.params.H)


	elif method == "RNNPTONE":
		#random.seed(10)
		L0 = zeros((vocabsize, hdim)) # replace with random init, 
					      # or do in RNNLM.__init__()
		model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

		h0_train = toONE(pickle.load(open('data/h0_train', 'rb')))
		h0_test = toONE(pickle.load(open('data/h0_test', 'rb')))
		idx = epochiter(len(Y_train), nepoch)
		model.train_sgd_rnnpt(X = X_train, y = Y_train, h0 = h0_train, idxiter = idx, printevery = 500, costevery = 500)

		dev_loss = model.compute_mean_loss(X_dev, Y_dev, h0_test)
		if not os.path.exists("model/" + method):
			os.makedirs("model/" + method)

		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		save("model/" + method + "/rnnlm.L.npy", model.sparams.L)
		save("model/" + method + "/rnnlm.U.npy", model.params.U)
		save("model/" + method + "/rnnlm.H.npy", model.params.H)
