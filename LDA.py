from gensim import *
import pandas as pd
import re
import os

def clean_corpus(statuses,s_words):
	df = {}
	tf = {}
	set_df = set()
	set_tf = set()
	tokenized = []
	for count, status in enumerate(statuses):
		status_list = set()
		##Remove non alphanumerics
		status = re.sub(r'[^a-zA-Z\s]', '', status)
		status = re.sub(r'([a-z])\1\1+', r'\1', status)
		for token in status.split():
			tok = token.lower()
			if not tok in s_words and not has_num(tok):
				if not tok in set_df:
					df[tok] = 0
					set_df.add(tok)
				if not tok in set_tf:
					tf[tok] = 0
					set_tf.add(tok)
				if tok not in status_list:
					df[tok] += 1
				tf[tok] += 1
				status_list.add(tok)
		tokenized.append(list(status_list))
		#print count, '   ', status
	tokenized_clean = clean_unicode(tokenized)
	print "DATA CLEANED"
	dictionary = corpora.Dictionary(tokenized_clean)
	print "Dictionary Made"
	corpus = [dictionary.doc2bow(token) for token in tokenized_clean]
	return corpus,df,dictionary,tf


if __name__ == '__main__':

	user_status = pd.read_csv(os.path.join('data', 'sample_status'), sep = ',', escapechar = '/', quotechar='"', error_bad_lines = False)
	user_per = pd.read_csv(os.path.join('data', 'sample_status'), sep = ',', escapechar = '\\', quotechar='"', error_bad_lines = False)


	agg = dict(list(Data.user_status_df.dropna().groupby('userid')))
185                 num_users = len(agg)
186                 #print '=================='
187                 #print list(agg['status_update'])[0:2]
188                 #print agg
189                 #print agg.columns
190                 user_index_to_id = {}
191                 counter = 0
192                 lis = []
193                 for i,row in agg.iteritems():
194                         #for status in row['status_update']:
195                         #if(counter < 5):
196                         #       print i
197                         #       print row
198                         user_index_to_id[counter] = i
199                         counter +=1
200                         lis.append(" ".join(list(row['status_update'])))
201                         #lis.append(status)
202                         #print row['status_update']
203                 print 'Number of users is ', num_users
204                 return lis,user_index_to_id



 def run_analysis_on_LDA_status():
 278         statuses,index_to_user_id = Data.get_status_corpus()
 279         s_words = set()
 280         corpus,df,dictionary,tf = clean_corpus(statuses,s_words)
 281         freq_words = get_freq_words(df,tf, len(statuses))
 282         for word in freq_words:
 283                 s_words.add(word)
 284         corpus,df,dictionary, tf = clean_corpus(statuses,s_words)
 285         print "DICTIONARY MDE"
 286         alphas = (0.01, 0.1, 1, 10)
 287         for a in alphas:
 288                 num_topics_to_perplexity = {}
 289                 lis = list(range(50))
 290                 results = multiprocessing.Pool(10).map(func_mapper, itertools.izip(itertools.repeat((corpus, a)), lis))
 291                 w = csv.writer(open(datapath+'perplexity_scores_' + str((int)(100*a)) + '.csv', "w"))
 292                 for result in results:
 293                         w.writerow([result[0], result[1]])


def get_freq_words(df,tf,n_docs, lda=[], dictionary = {}):
  93         print "REMOVING WORDS"
  94         to_remove = set()
  95         for word in df.keys():
  96                 if df[word] > (n_docs*.1):
  97                         to_remove.add(word)
  98                 if df[word] < 10:
  99                         to_remove.add(word)
 100         temp = []
 101         for word in tf.keys():
 102                 heapq.heappush(temp, (-tf[word], word))
 103         for i in range(500):
 104                 to_remove.add(heapq.heappop(temp)[1])
 105 
 106         for key,val in dictionary.iteritems():
 107                 prob = lda[[[(key, 1)]]]
 108                 #print prob
 109                 if len(prob)<1:
 110                         to_remove.add(val)
 111                 if len(prob)>3:
 112                         to_remove.add(val)
 113         print to_remove
 114         return to_remove
