import sys, os
from numpy import *
from rnnlm import RNNLM, RNNPT
from data_utils import utils as du
import pandas as pd
from wordGeneration import toONE, adjust_loss
import cPickle as pickle



def seq_to_words(seq):
	return [num_to_word[s] for s in seq]

def bleu(sentence, reference, N):
	sentence = sentence[1:-1]
	reference = reference[:-1]
	matched = 0
	total = 0
	for n in xrange(1, N + 1):
		for s in xrange(len(sentence) - n + 1):
			for r in xrange(len(reference) - n + 1):
				all_matched = True
				for offset in xrange(n):
					if (sentence[s + offset] != reference[r + offset]) or sentence[s + offset] == 3 or reference[r + offset] == 3: #UUUNKKK
						all_matched = False
						break
				if all_matched:
					matched += 1
					break
			total += 1
	if total == 0:
		return 0
	return float(matched) / total


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
	method = "RNNPTONE"
	#evaluation = "loss"
	#evaluation = "zero"
	evaluation = "three"

	printTag = method + " " + evaluation + " "
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
			gram1 = zeros(len(Y_dev))
			gram2 = zeros(len(Y_dev))
			gram3 = zeros(len(Y_dev))
			for i in xrange(len(Y_dev)):
				if i % 1000 == 0:
					print printTag + str(i)
				seq, J = model.generate_sequence(word_to_num["<s>"], 
										 word_to_num["</s>"], 
										 maxlen=100)
				gram1[i] = bleu(seq, Y_dev[i], 1)
				gram2[i] = bleu(seq, Y_dev[i], 2)	
				gram3[i] = bleu(seq, Y_dev[i], 3)
			print printTag
			print mean(gram1)
			print mean(gram2)
			print mean(gram3)
		elif evaluation == "three":
			gram1 = zeros(len(Y_dev))
			gram2 = zeros(len(Y_dev))
			gram3 = zeros(len(Y_dev))
			for i in xrange(len(Y_dev)):
				if i % 1000 == 0:
					print printTag + str(i)
				if len(Y_dev[i]) >= 3:
					words = Y_dev[i][:3]
				else:
					words = Y_dev[i]
				seq, J = model.generate_sequence(word_to_num["<s>"], 
										 word_to_num["</s>"], 
										 maxlen=100, words = words)
				gram1[i] = bleu(seq, Y_dev[i], 1)
				gram2[i] = bleu(seq, Y_dev[i], 2)	
				gram3[i] = bleu(seq, Y_dev[i], 3)
			print printTag
			print mean(gram1)
			print mean(gram2)
			print mean(gram3)
			#print J
			#print " ".join(seq_to_words(seq))
		
		


	elif method == "RNNPT" or method == "RNNPTONE":
		
		model = RNNPT(L0, U0 = L0, alpha=0.1,  bptt=3)

		if method == "RNNPT":
			h0_train = pickle.load(open('data/h0_train', 'rb'))
			h0_test = pickle.load(open('data/h0_test', 'rb'))
		elif method == "RNNPTONE":
			h0_train = toONE(pickle.load(open('data/h0_train', 'rb')))
			h0_test = toONE(pickle.load(open('data/h0_test', 'rb')))
		model.sparams.L = load("model/" + method + "/rnnlm.L.npy")
		model.params.U = load("model/" + method + "/rnnlm.U.npy")
		model.params.H = load("model/" + method + "/rnnlm.H.npy" )
		if evaluation == "loss":
			dev_loss = model.compute_mean_loss(X_dev, Y_dev, h0_test)
			print "Unadjusted: %.03f" % exp(dev_loss)
			print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
		elif evaluation == "zero":
			gram1 = zeros(len(Y_dev))
			gram2 = zeros(len(Y_dev))
			gram3 = zeros(len(Y_dev))
			for i in xrange(len(Y_dev)):
				if i % 1000 == 0:
					print printTag + str(i)
				seq, J = model.generate_sequence(word_to_num["<s>"], 
										 word_to_num["</s>"], 
										 h0_test[i], maxlen=100)
				gram1[i] = bleu(seq, Y_dev[i], 1)
				gram2[i] = bleu(seq, Y_dev[i], 2)	
				gram3[i] = bleu(seq, Y_dev[i], 3)
				if len(seq) >= 10 and gram1[i] > 0.5:
					print " ".join(seq_to_words(seq))
					print " ".join(seq_to_words(Y_dev[i]))
			print printTag

			print mean(gram1)
			print mean(gram2)
			print mean(gram3)
		elif evaluation == "three":
			gram1 = zeros(len(Y_dev))
			gram2 = zeros(len(Y_dev))
			gram3 = zeros(len(Y_dev))
			for i in xrange(len(Y_dev)):
				if i % 1000 == 0:
					print printTag + str(i)
				if len(Y_dev[i]) >= 3:
					words = Y_dev[i][:3]
				else:
					words = Y_dev[i]
				seq, J = model.generate_sequence(word_to_num["<s>"], 
										 word_to_num["</s>"], 
										 h0_test[i], maxlen=100, words = words)
				gram1[i] = bleu(seq, Y_dev[i], 1)
				gram2[i] = bleu(seq, Y_dev[i], 2)	
				gram3[i] = bleu(seq, Y_dev[i], 3)
				if len(seq) >= 10 and gram1[i] > 0.5:
					print " ".join(seq_to_words(seq))
					print " ".join(seq_to_words(Y_dev[i]))
			print printTag

			print mean(gram1)
			print mean(gram2)
			print mean(gram3)


