import sys, os
from numpy import *
from rnnlm import RNNLM
from data_utils import utils as du
import pandas as pd



def seq_to_words(seq):
    return [num_to_word[s] for s in seq]

# Load the vocabulary
vocab = pd.read_table("data/dictionary", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )
# Choose how many top words to keep
vocabsize = len(vocab)
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)


hdim = 45
L0 = zeros((vocabsize, hdim))
model = RNNLM(L0, U0 = L0, alpha=0.1, rseed=10, bptt=4)
model.params.U=load("model/rnnlm.U.npy")
model.sparams.L=load("model/rnnlm.L.npy")
#L = load("model/rnnlm.L.npy")
#print L.shape
model.params.H=load("model/rnnlm.H.npy")
seq, J = model.generate_sequence(word_to_num["<s>"], 
                                 word_to_num["</s>"], 
                                 maxlen=100)
print J
# print seq
print " ".join(seq_to_words(seq))
