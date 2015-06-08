import sys, os
from numpy import *
from rnnlm import RNNLM, RNNPT
from data_utils import utils as du
import pandas as pd
from wordGeneration import toONE, adjust_loss



def seq_to_words(seq):
	return [num_to_word[s] for s in seq]



if __name__ == "__main__":	
	# Load the vocabulary
	vocab = pd.read_table("data/dictionary", header=None, sep="\s+",
						 index_col=0, names=['count', 'freq'], )
	# Choose how many top words to keep
	vocabsize = len(vocab)
	num_to_word = dict(enumerate(vocab.index[:vocabsize]))
	word_to_num = du.invert_dict(num_to_word)


	docs = du.load_dataset('data/rnn_input_test')
	S_train = du.docs_to_indices(docs, word_to_num)
	X_dev, Y_dev = du.seqs_to_lmXY(S_train)

	hdim = 40
	L0 = zeros((vocabsize, hdim))
	fraction_lost = 0.07923163705
	method = RNNLM
	evaluation = "loss"
	#evaluation = "zero"
	#evaluation = "three"
 	if method == "RNNLM":
		
		model = RNNLM(L0, U0 = L0, alpha=0.1,  bptt=3)
		model.sparams.L = load("model/" + method + "/rnnlm.L.npy")
		model.params.U = load("model/" + method + "/rnnlm.U.npy")
		model.params.H = load("model/" + method + "/rnnlm.H.npy" )

		if evaluation == "loss":
			dev_loss = model.compute_mean_loss(X_dev, Y_dev)
			print "Unadjusted: %.03f" % exp(dev_loss)
			print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		elif evaluation == "zero":
			seq, J = model.generate_sequence(word_to_num["<s>"], 
									 word_to_num["</s>"], 
									 maxlen=100)
			print J
			print " ".join(seq_to_words(seq))
		
		print "RNNLM"


	elif method == "RNNPT":
		
		model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

		h0_train = pickle.load(open('data/h0_train', 'rb'))
		h0_test = pickle.load(open('data/h0_test', 'rb'))
		model.sparams.L = load("model/" + method + "/rnnlm.L.npy")
		model.params.U = load("model/" + method + "/rnnlm.U.npy")
		model.params.H = load("model/" + method + "/rnnlm.H.npy" )
		dev_loss = model.compute_mean_loss(X_dev, Y_dev, h0_test)
		

		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		print "RNNPT"


	elif method == "RNNPTONE":
		model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

		h0_train = toONE(pickle.load(open('data/h0_train', 'rb')))
		h0_test = toONE(pickle.load(open('data/h0_test', 'rb')))
		model.sparams.L = load("model/" + method + "/rnnlm.L.npy")
		model.params.U = load("model/" + method + "/rnnlm.U.npy")
		model.params.H = load("model/" + method + "/rnnlm.H.npy" )
		dev_loss = model.compute_mean_loss(X_dev, Y_dev, h0_test)
		print "Unadjusted: %.03f" % exp(dev_loss)
		print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		print "RNNPTONE"


