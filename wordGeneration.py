import sys, os
from numpy import *
from matplotlib.pyplot import *

matplotlib.rcParams['savefig.dpi'] = 100

from rnnlm import RNNLM
from data_utils import utils as du
import pandas as pd

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

docs = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs, word_to_num)
X_train, Y_train = du.seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs, word_to_num)
X_dev, Y_dev = du.seqs_to_lmXY(S_dev)

# Load the test set (final evaluation only)
docs = du.load_dataset('data/lm/ptb-test.txt')
S_test = du.docs_to_indices(docs, word_to_num)
X_test, Y_test = du.seqs_to_lmXY(S_test)

hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = zeros((vocabsize, hdim)) # replace with random init, 
                              # or do in RNNLM.__init__()
model = RNNLM(L0, U0 = L0, alpha=0.1, rseed=10, bptt=4)

ntrain = len(Y_train)
nepoch = 5
N = nepoch * len(Y_train)
k = 5 # minibatch size
random.seed(10)
idx=[]
print X_train.size
for i in range(N/k):
    idx.append(random.choice(len(Y_train),k))
model.train_sgd(X_train,Y_train,idx,None,10000,10000,None)

dev_loss = model.compute_mean_loss(X_dev, Y_dev)
def adjust_loss(loss, funk):
    return (loss + funk * log(funk))/(1 - funk)
print "Unadjusted: %.03f" % exp(dev_loss)
print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))
