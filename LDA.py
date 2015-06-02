import gensim
import pandas as pd
import re
import os
import multiprocessing
import math
import logging

def clean_corpus(statuses,s_words):
	pass
	# df = {}
	# tf = {}
	# set_df = set()
	# set_tf = set()
	# tokenized = []
	# for count, status in enumerate(statuses):
	# 	status_list = set()
	# 	##Remove non alphanumerics
	# 	status = re.sub(r'[^a-zA-Z\s]', '', status)
	# 	status = re.sub(r'([a-z])\1\1+', r'\1', status)
	# 	for token in status.split():
	# 		tok = token.lower()
	# 		if not tok in s_words and not has_num(tok):
	# 			if not tok in set_df:
	# 				df[tok] = 0
	# 				set_df.add(tok)
	# 			if not tok in set_tf:
	# 				tf[tok] = 0
	# 				set_tf.add(tok)
	# 			if tok not in status_list:
	# 				df[tok] += 1
	# 			tf[tok] += 1
	# 			status_list.add(tok)
	# 	tokenized.append(list(status_list))
	# 	#print count, '   ', status
	# tokenized_clean = clean_unicode(tokenized)
	# print "DATA CLEANED"
	# dictionary = corpora.Dictionary(tokenized_clean)
	# print "Dictionary Made"
	# corpus = [dictionary.doc2bow(token) for token in tokenized_clean]
	# return corpus,df,dictionary,tf
	




	


def get_status_corpus(user_status):
	# agg = dict(list(user_status.dropna().groupby('userid')))
	# num_users = len(agg)

	# user_index_to_id = {}
	# counter = 0
	# lis = []
	# for i,row in agg.iteritems():

	# 	user_index_to_id[counter] = i
	# 	counter +=1
	# 	lis.append(" ".join(list(row['status_update'])))
	     
	# print 'Number of users is ', num_users
	# return lis,user_index_to_id
	statuses =  list(user_status['status_update'])

	corpus = []
	for i in xrange(len(statuses)):
		status = statuses[i]
		if isinstance(status, str) :
			status = re.sub(r'[^a-zA-Z\s]', '', status)
			status = re.sub(r'([a-z])\1\1+', r'\1', status)
			status = status.lower()
			corpus.append(status)
		# else:
		# 	print status


	stoplist = set('for a of the and to in'.split())
	corpus = [[word for word in document.split() if word not in stoplist]
		for document in corpus]
	print 'tokenized'

	from collections import defaultdict
	frequency = defaultdict(int)
	for text in corpus:
		for token in text:
			frequency[token] += 1

	corpus = [[token for token in text if frequency[token] > 10 and frequency[token] < 5000]
	 for text in corpus]
	
	print 'remove unfrequent and frequent words'

	dictionary = gensim.corpora.Dictionary(corpus)
	dictionary.save('data/lda.dict')
	print 'dictionary'

	corpus = [dictionary.doc2bow(text) for text in corpus]
	gensim.corpora.MmCorpus.serialize('data/lda.mm', corpus)
	print 'text corpus'
	return corpus



def run_analysis_on_LDA_status(preprocess=False):
	if preprocess == True:
		user_status = pd.read_csv(os.path.join('data', 'sample_status'), sep = ',')#, escapechar = '/', quotechar='"')
		user_per = pd.read_csv(os.path.join('data', 'sample_personality'), sep = ',')#, escapechar = '\\', quotechar='"', error_bad_lines = False)
		statuses = get_status_corpus(user_status)
	else:	

		id2word = gensim.corpora.Dictionary.load('data/lda.dict')
		mm = gensim.corpora.MmCorpus('data/lda.mm')
		print mm
		#lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=20, update_every=1, chunksize=1000, passes=1)
		lda = gensim.models.LdaMulticore(corpus=mm, num_topics=20, id2word=id2word, workers=8)
		lda.save('model/lda.model')
		lda.show_topics(num_topics = 20)

	# freq_words = get_freq_words(df,tf, len(statuses))
	# for word in freq_words:
	# 	s_words.add(word)
	# corpus,df,dictionary, tf = clean_corpus(statuses,s_words)
	# print "DICTIONARY MDE"
	# alphas = (0.01, 0.1, 1, 10)
	# for a in alphas:
	# 	num_topics_to_perplexity = {}
	# 	lis = list(range(50))
	# 	results = multiprocessing.Pool(10).map(func_mapper, itertools.izip(itertools.repeat((corpus, a)), lis))
	# 	w = csv.writer(open(datapath+'perplexity_scores_' + str((int)(100*a)) + '.csv', "w"))
	# 	for result in results:
	# 		w.writerow([result[0], result[1]])


def get_freq_words(df,tf,n_docs, lda=[], dictionary = {}):
	print "REMOVING WORDS"
	to_remove = set()
	for word in df.keys():
		if df[word] > (n_docs*.1):
			to_remove.add(word)
		if df[word] < 10:
			to_remove.add(word)
	temp = []
	for word in tf.keys():
		heapq.heappush(temp, (-tf[word], word))
	for i in range(500):
		to_remove.add(heapq.heappop(temp)[1])

	for key,val in dictionary.iteritems():
		prob = lda[[[(key, 1)]]]

		if len(prob)<1:
			to_remove.add(val)
		if len(prob)>3:
			to_remove.add(val)
	print to_remove
	return to_remove


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	run_analysis_on_LDA_status(preprocess=True)
