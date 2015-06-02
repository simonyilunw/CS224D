import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from scipy import sparse
from scipy import stats
import csv
from scipy import io
import pickle

class Data:
	##data frames
	user_like_df = None
	like_text_df = None
	diad_list_df = None
	user_demog_df = None
	user_address_df = None
	user_personality_df = None
	user_status_df = None
	location_id_to_description = {}

	##path = './'
	path = '/dfs/scratch1/fb_personality_research/data/'
	#outpath = '/dfs/scratch1/fb_personality_research/temp/user_like_50/'
	outpath = 'user_like/'
	##from id, return list of all likes
	@staticmethod
	def get_user_like_text(id):
		like_ids = Data.get_like_ids(id)
		like_text = Data.get_like_text(like_ids)
		return like_text

	#Reads the data into dataframes and removes quotes from the dataframe names (causes problem in the diads function)
	@staticmethod
	def read_df(f_name):
		if f_name == 'fb_like.csv':
			temp_df =  pd.read_csv(Data.path+f_name,sep = ',',escapechar = '\\',quotechar = '\n',error_bad_lines = False)	
			return temp_df
		if f_name == 'user_status.csv':
			temp_df = pd.read_csv(Data.path+f_name, sep = ',', escapechar = '/', quotechar='"', error_bad_lines = False)	
		else:
			temp_df = pd.read_csv(Data.path+f_name,sep = ',',escapechar = '\\',quotechar = '"',error_bad_lines = False)
		quoted_names = ",".join(temp_df.columns)
		temp_df.columns = (quoted_names.replace("\"", "")).split(",")
		if f_name == 'address.csv':
			id_to_loc_df = pd.read_csv(Data.path+'location_dict.csv', sep=',', quotechar='"')
			state_list = pickle.load(open(Data.path+'US_States.pickle', 'rb'))
			id_to_loc_df = id_to_loc_df.join(id_to_loc_df['name'].apply(lambda x: pd.Series(x.split(', '))))
			id_to_loc_df[2][id_to_loc_df[1].isin(state_list)] = ['United States']*id_to_loc_df[id_to_loc_df[1].isin(state_list)].shape[0]
			#id_to_loc_df = id_to_loc_df.rename(columns={0: "current_location_city", 1: "current_location_state", 2: "current_location_country"})
			id_to_loc_df = id_to_loc_df[['location_id', 0, 1, 2]]
			temp_df = temp_df.merge(id_to_loc_df, on='location_id', how='left')
			temp_df['current_location_city'].fillna(temp_df[0], inplace=True)
			temp_df['current_location_state'].fillna(temp_df[1], inplace=True)
			temp_df['current_location_country'].fillna(temp_df[2], inplace=True)

		return temp_df
				
	
	## get ids for likes
	@staticmethod	
	def get_like_ids(id):
		if Data.user_like_df is None:
			Data.user_like_df = Data.read_df('user_like.csv')	
		relevant_ids = Data.user_like_df[Data.user_like_df['userid'] == id]['like_id']
		#print(relevant_ids)
		return list(relevant_ids)

	##get text from a set of like ids
	@staticmethod	
	def get_like_text(like_ids):
		if Data.like_text_df is None:
			Data.like_text_df = Data.read_df('fb_like.csv')	
		relevant_text = Data.like_text_df[Data.like_text_df['like_id'].isin(like_ids)]
		return relevant_text	

	#get list of user ids from like id
	@staticmethod
	def get_user_ids(like_id):
		if Data.user_like_df is None:
			Data.user_like_df = Data.read_df('user_like.csv')
		user_ids = Data.user_like_df[Data.user_like_df['like_id']==like_id]['userid']
		return list(user_ids)
	

	@staticmethod
	def get_user_like_stats(user_ids = [],use_all_users = True):
		if Data.user_like_df is None:
			Data.user_like_df = Data.read_df('user_like.csv')
		if not use_all_users:
			df_user_stats = Data.user_like_df[Data.user_like_df['userid'] in user_id]
		else:
			df_user_stats = Data.user_like_df 
		like_dist =  df_user_stats.groupby('like_id').size() 
		user_dist = df_user_stats.groupby('userid').size() 	
		print "LIKE AVERAGE: " + str(like_dist.mean())
		print "USER AVERAGE: " + str(user_dist.mean())
		print "LIKE STD: " + str(like_dist.std())
		print "USER STD: " + str(user_dist.std())
		print "LIKE MAX: " + str(like_dist.max())
		print "USER MAX: " + str(user_dist.max())
		ids =  like_dist[like_dist > 100]
	
		##PLOT HISTOGRAMS
		plt.figure()
		plt.subplot(222)
		##sub_like,axis = plt.subplots()
		plt.title("like dist")
		like_dist.hist(bins = range(50, 1000, 10))
		plt.subplot(221)
		plt.title("user dist")
		user_dist.hist(bins = range(2000))		
		plt.show()		
		return like_dist,user_dist	
	
	#generate user-like matrix from data file
	#Returns dataframe of users, likes and unique indices
	@staticmethod
	def generate_user_like_matrix(user_ids = [], use_all_users = True, like_ids = []):
		if Data.user_like_df is None:
                        Data.user_like_df = Data.read_df('user_like.csv')
		if not use_all_users:
			users_unique_df = pd.DataFrame(user_ids, columns = ['userid'])
		else:
			users_unique_df = pd.DataFrame(Data.user_like_df['userid'].drop_duplicates())
			users_unique_df.columns = ['userid']
		likes_unique_df = pd.DataFrame(Data.user_like_df['like_id'].drop_duplicates())
		if like_ids:
			likes_unique_df = likes_unique_df[likes_unique_df['like_id'].isin(like_ids)]
		likes_unique_df.columns = ['like_id']
		
		likes_unique_df['index_likes'] = range(likes_unique_df.shape[0])
		users_unique_df['index_users'] = range(users_unique_df.shape[0])
      		merged_df = pd.DataFrame.merge(pd.DataFrame.merge(Data.user_like_df, users_unique_df, on=['userid']), likes_unique_df, on = ['like_id'])
		user_id_to_index = dict(zip(list(merged_df['index_users']),list(merged_df['userid'])))	
		like_id_to_index = dict(zip(list(merged_df['index_likes']),list(merged_df['like_id'])))	
		sparse_mat = sparse.csr_matrix(([1]*merged_df.shape[0],(list(merged_df['index_users']),list(merged_df['index_likes']))),shape = (users_unique_df.shape[0],likes_unique_df.shape[0]) )
		print sparse_mat.shape
		print len(user_id_to_index)
		print len(like_id_to_index)
		return merged_df,user_id_to_index,like_id_to_index,sparse_mat
	
	@staticmethod	
	def print_full(x):
		pd.set_option('display.max_rows',min(5000,len(x)))
		print(x)
		pd.reset_option('display.max_rows')

	@staticmethod
	def get_country_stats(user_ids = [],use_all_users = True):
		df_country = Data.user_address_df	
		df_country = df_country.fillna("NOWHERE")	
		country_counts = df_country.groupby('current_location_country').size()
		country_counts.sort()
		Data.print_full(country_counts)
		country_counts = df_country[df_country['current_location_country']== 'United States'].groupby(['current_location_country','current_location_state']).size()
		country_counts.sort()
		Data.print_full(country_counts)
	
	@staticmethod 
	def get_american_ids():
		if Data.user_address_df is None:
			Data.user_address_df = Data.read_df('address.csv') 	
		return list(Data.user_address_df[Data.user_address_df['current_location_country']=='United States']['userid'])	
		

	@staticmethod 
	def get_status_corpus():		
		#user_ids = Data.read_dict_from_file(Data.outpath + "user_index.csv").values()
		#print user_ids
		if Data.user_status_df is None:
			##Data.user_status_df = Data.read_df('user_status.csv')
			Data.user_status_df = Data.read_df('user_status.csv')
			##Data.user_status_df = pd.load( './user_local_status.csv')
			##Data.user_status_df = Data.read_df('user_status.csv')
		#Data.user_status_df = Data.user_status_df[Data.user_status_df['userid'].isin(user_ids)]  
		Data.user_status_df.save('./user_local_status.csv') 
		##return Data.user_status_df.groupby('userid'),Data.user_status_df
		print "DATA READ IN"
		#print Data.user_status_df.head()
		#print Data.user_status_df.shape
		#print Data.user_status_df.columns
		#print Data.user_status_df['status_update']
		##return Data.user_status_df
		agg = dict(list(Data.user_status_df.dropna().groupby('userid')))
		num_users = len(agg)
		#print '=================='
		#print list(agg['status_update'])[0:2]
		#print agg
		#print agg.columns
		user_index_to_id = {}
		counter = 0
		lis = []
		for i,row in agg.iteritems():
			#for status in row['status_update']:
			#if(counter < 5):
			#	print i
			#	print row
			user_index_to_id[counter] = i
			counter +=1
			lis.append(" ".join(list(row['status_update'])))
			#lis.append(status)
			#print row['status_update']
		print 'Number of users is ', num_users
		return lis,user_index_to_id
		
	

	@staticmethod	
	def get_data_stats():
	
		if Data.user_like_df is None:
			Data.user_like_df = Data.read_df('user_like.csv') 
		if Data.user_personality_df is None:
			Data.user_personality_df = Data.read_df('big5.csv') 
		if Data.user_demog_df is None:
			Data.user_demog_df = Data.read_df('demog.csv') 
		if Data.user_address_df is None:
			Data.user_address_df = Data.read_df('address.csv') 
		if Data.diad_list_df is None:
			Data.diad_list_df = Data.read_df('fb_friendship.csv')
			Data.index_diad_df() 
		
		##Data.get_user_like_stats([], True)
		ul = Data.user_like_df.groupby('like_id').size()
		l_50 = ul[ul>50]
		l_100 = ul[ul>100]
		ul = Data.user_like_df.groupby('userid').size()
		temp_50 = list(ul[ul>50].keys())
		temp_100 = list(ul[ul>100].keys())
		u_50 = pd.DataFrame(temp_50, columns = ['userid'])
		u_100 = pd.DataFrame(temp_100, columns=['userid'])
	
		c = Data.user_address_df[Data.user_address_df['current_location_country']=='United States']
		print 'c %d' % c.shape[0]
		ch = c[c['hometown_location_country']=='United States']
		print 'ch %d' % ch.shape[0]
		chs = ch[pd.notnull(ch['current_location_state'])]
		print 'chs %d' % chs.shape[0]
		chsw = chs[pd.notnull(chs['hometown_location_state'])]['userid']	
		print 'chsw %d' % chsw.shape[0]

		g = Data.user_demog_df[pd.notnull(Data.user_demog_df['gender'])]
		print 'g %d' % g.shape[0]
		ga = g[pd.notnull(g['age'])]
		print 'ga %d' % ga.shape[0]
		gat = ga[pd.notnull(ga['timezone'])]
		gatr = gat[pd.notnull(gat['relationship_status'])]
		gatrn = gatr[pd.notnull(gatr['network_size'])]['userid']
		
		b = Data.user_personality_df['userid']
		print 'Number of users in personality database%d' % b.count()

		print 'Number of unique diads ', Data.diad_list_df.shape
		#print 'Test value', len(u_50), set(u_50['userid'])
		u_50_gatrn = set(u_50['userid']) & set(gatrn)	
		print 'u_50_gatrn %d' % len(list(u_50_gatrn))
 		u_100_gatrn = set(u_100['userid']) & set(gatrn)

		print 'u_100_gatrn %d' % len(u_100_gatrn)
		u_50_chsw =  (set(u_50['userid']) & set(chsw))

		print 'u_50_chsw %d' % len(u_50_chsw) 
 		u_100_chsw = (set(u_100['userid']) & set(chsw))

		print 'u_100_chsw %d' % len(u_100_chsw)
		u_50_ch = (set(u_50['userid']) & set(ch['userid']))

		print 'u_50_ch %d' % len(u_50_ch)
		u_100_ch =  (set(u_100['userid']) & set(ch['userid']))

		print 'u_100_ch %d' % len(u_100_ch)
		u_50_ga = set(u_50['userid']) & set(ga['userid'])
		#print u_50_ga
		print 'u_50_ga %d' % len(u_50_ga)
 		u_100_ga = set(u_100['userid']) & set(ga['userid'])

		print 'u_100_ga %d' % len(u_100_ga)
		
		u_50_b =  (set(u_50['userid']) & set(b))

		print 'u_50_b %d' % len(u_50_b)
		u_100_b =  (set(u_100['userid']) & set(b))



		print 'u_100_b %d' % len(u_100_b)
		print 'u_50_b_ch %d' % len(u_50_b & u_50_ch)
		print 'u_50_b_chsw %d' % len(u_50_b & u_50_chsw)
		print 'u_50_b_gatrn %d' % len(u_50_b & u_50_gatrn)
		print 'u_50_b_ga %d' % len(u_50_b & u_50_ga)
		
		print 'u_100_b_ch %d' % len(u_100_b & u_100_ch)
		print 'u_100_b_chsw %d' % len(u_100_b & u_100_chsw)
		print 'u_100_b_gatrn %d' % len(u_100_b & u_100_gatrn)
		print 'u_100_b_ga %d' % len(u_100_b & u_100_ga)

		print 'u_50_ch_gatrn %d' % len(u_50_ch & u_50_gatrn)
		print 'u_50_chsw_gatrn %d' % len(u_50_chsw & u_50_gatrn)
		print 'u_50_ch_ga %d' % len( u_50_ch & u_50_ga)
		print 'u_50_chsw_ga %d' % len(u_50_chsw & u_50_ga)
 
		print 'u_100_ch_gatrn %d' % len( u_100_ch & u_100_gatrn)
		print 'u_100_chsw_gatrn %d' % len(u_100_chsw & u_100_gatrn)
		print 'u_100_ch_ga %d' % len(u_100_ch & u_100_ga)
		print 'u_100_chsw_ga %d' % len(u_100_chsw & u_100_ga)
 

		print 'u_50_b_ch_gatrn %d' % len(u_50_b & u_50_ch & u_50_gatrn)
		print 'u_50_b_chsw_gatrn %d' % len(u_50_b & u_50_chsw & u_50_gatrn)
		print 'u_50_b_ch_ga %d' % len(u_50_b & u_50_ch & u_50_ga)
		print 'u_50_b_chsw_ga %d' % len(u_50_b & u_50_chsw & u_50_ga)
 
		print 'u_100_b_ch_gatrn %d' % len(u_100_b & u_100_ch & u_100_gatrn)
		print 'u_100_b_chsw_gatrn %d' % len(u_100_b & u_100_chsw & u_100_gatrn)
		print 'u_100_b_ch_ga %d' % len(u_100_b & u_100_ch & u_100_ga)
		print 'u_100_b_chsw_ga %d' % len(u_100_b & u_100_chsw & u_100_ga)
 
	
		print 'd_u_50 %d' % Data.get_reduced_diads(u_50).shape
		print 'd_u_100 %d' % Data.get_reduced_diads(u_100).shape
			
		print 'd_u_50_gatrn %d' %  Data.get_reduced_diads(u_50, {}, u_50_gatrn).shape

		print 'd_u_100_gatrn %d' % Data.get_reduced_diads(u_100, {}, u_100_gatrn).shape

		print 'd_u_50_chsw %d' %  Data.get_reduced_diads(u_50, {}, u_50_chsw).shape

		print 'd_u_100_chsw %d' %  Data.get_reduced_diads(u_100, {}, u_100_chsw).shape

		print 'd_u_50_ch %d' %  Data.get_reduced_diads(u_50, {}, u_50_ch).shape

		print 'd_u_100_ch %d' % Data.get_reduced_diads(u_100, {}, u_100_ch).shape

		print 'd_u_50_b %d' % Data.get_reduced_diads(u_50, {}, u_50_b).shape

		print 'd_u_100_b %d' %  Data.get_reduced_diads(u_100, {}, u_100_b).shape

		print 'd_u_50_b_ch %d' %  Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_ch))).shape


		print 'd_u_50_b_gatrn %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_gatrn))).shape
		print 'd_u_50_b_ga %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_ga))).shape
	
		print 'd_u_100_b_ch %d' %  Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_ch))).shape

		print 'd_u_100_b_chsw %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_chsw))).shape
		print 'd_u_100_b_gatrn %d' %  Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_gatrn))).shape

		print 'd_u_100_b_ga %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_ga))).shape

		print 'd_u_50_ch_gatrn %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_ch) & set(u_50_gatrn))).shape

		print 'd_u_50_chsw_gatrn %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_chsw) & set(u_50_gatrn))).shape

		print 'd_u_50_ch_ga %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_ch) & set(u_50_ga))).shape


		print 'd_u_50_chsw_ga %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_chsw) & set(u_50_ga))).shape

		print 'd_u_100_ch_gatrn %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_ch) & set(u_100_gatrn))).shape


		print 'd_u_100_chsw_gatrn %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_chsw) & set(u_100_gatrn))).shape
		print 'd_u_100_ch_ga %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_ch) & set(u_100_ga))).shape
		print 'd_u_100_chsw_ga %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_chsw) & set(u_100_ga))).shape
 


		print 'd_u_50_b_ch_gatrn %d' %  Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_ch) & set(u_50_gatrn))).shape

		print 'd_u_50_b_chsw_gatrn %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_chsw) & set(u_50_gatrn))).shape
		print 'd_u_50_b_ch_ga %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_ch) & set(u_50_ga))).shape
		print 'd_u_50_b_chsw_ga %d' % Data.get_reduced_diads(u_50, {}, list(set(u_50_b) & set(u_50_chsw) & set(u_50_ga))).shape
 
		print 'd_u_100_b_ch_gatrn %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_ch) & set(u_100_gatrn))).shape
		print 'd_u_100_b_chsw_gatrn %d' %  Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_chsw) & set(u_100_gatrn))).shape

		print 'd_u_100_b_ch_ga %d' %  Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_ch) & set(u_100_ga))).shape

		print 'd_u_100_b_chsw_ga %d' % Data.get_reduced_diads(u_100, {}, list(set(u_100_b) & set(u_100_chsw) & set(u_100_ga))).shape

 
		
		#Get stats for number of users by country
		#Data.get_country_stats([], True)
		
		#Get stats for number of diads with data (personality, demog, like, status, views and intersections) for both users
		
	#Function to index diads -- assign a unique identifier to each diad in the diad dataframe, removes quotes from the column names
	@staticmethod
	def index_diad_df():
		Data.diad_list_df['index_diads'] = range(Data.diad_list_df.shape[0])
		#print Data.diad_list_df.columns
		
	#Function that takes in a dataframe and a dictionary of column names to values as input and returns the indexes of diads that have both the users that agree with the dictionary
	@staticmethod
	def get_reduced_diads(test_df, column_to_value = {}, list_ids = []):
		if Data.diad_list_df is None:
			Data.diad_list_df = Data.read_df('fb_friendship.csv')
			Data.index_diad_df()
		#print Data.diad_list_df
		user_ids = []
		if len(list_ids) > 0:	
			test_df = test_df[test_df['userid'].isin(list_ids)]
			user_ids = list_ids
		for column_name in column_to_value.keys():	
			test_df = test_df[test_df[column_name]==column_to_value[column_name]]
		if len(user_ids) > 0:
			user_ids = list(set(user_ids) & set(test_df['userid']))
		else:
			user_ids = list(test_df['userid'])
		#print user_ids
		return pd.DataFrame.merge(Data.diad_list_df[Data.diad_list_df['friend1'].isin(user_ids)], Data.diad_list_df[Data.diad_list_df['friend2'].isin(user_ids)], on='index_diads')['index_diads']

	#Returns users with more than 50 likes and appearing in the intersection of user-status and living in the US
	@staticmethod
	def get_data_for_basic_model(use_all_users = False, n_likes = 50):
		#if Data.user_like_df is None:
		#	Data.user_like_df = Data.read_df('user_like.csv')
		#List of american users
		list_american_ids = Data.get_american_ids()

		#List of users in the user_status df
		Data.user_status_df = Data.read_df('user_status.csv')
		u_status = list(Data.user_status_df['userid'])
		personality = pd.read_csv(Data.path + 'big5.csv',sep = ',',escapechar = '\\',quotechar = '"',error_bad_lines = False)
		u_per = list(personality['userid'])

		u_50 = list(set(u_status) & set(u_per) & set(list_american_ids))
		temp = Data.user_status_df[Data.user_status_df['userid'].isin(u_50)]
		ul = temp.groupby('userid').size()
		u_50 = list(ul[ul>n_likes].keys())
		print 'Num. of users %d' % len(u_50)

		sample_status = Data.user_status_df[Data.user_status_df['userid'].isin(u_50)]	
		good_status = []
		for i, row in sample_status.iterrows():
			if len(str(row['status_update']).split()) > 20:
				good_status.append(i)

		sample_status = sample_status[sample_status.index.isin(good_status)]

		sample_status.to_csv('data/sample_status', quotechar = '"'  )
		print 'Num. of status %d' % len(sample_status)

		personality = personality[personality['userid'].isin(u_50)]
		personality.to_csv('data/sample_personality', quotechar = '"')

		
		#List of pages with more than 50 likes
		#ul = Data.user_like_df.groupby('like_id').size()
		#l_50 = list(ul[(ul>50)].keys())
		#List of users with more than 50 likes

		#Intersection of all the user lists
		#u_list = list(set(u_50) & set(u_status)&set(list_american_ids))
		
		#u_list = list(set(u_50)  & set(u_happy))
		#by yilun
		#temp = Data.user_like_df[Data.user_like_df['userid'].isin(u_list)]
		#ul = temp.groupby('like_id').size()
		#l_50 = list(ul[ul>10].keys())

		#temp = temp[temp['like_id'].isin(l_50)]
		#ul = temp.groupby('userid').size()
		#u_list = list(ul[ul>0].keys())
		
		#temp, user_to_index, like_to_index, user_like_matrix = Data.generate_user_like_matrix(u_list, False, l_50)

		#Saving the matrix in a file
		#np.savez(Data.outpath + 'user_like_matrix_' + str(n_likes),data=user_like_matrix.data, indices = user_like_matrix.indices, indptr = user_like_matrix.indptr, shape = user_like_matrix.shape)
		
		#Saving the user to index dictionary to a file
		#Data.write_dict_to_file(Data.outpath + "user_index_" + str(n_likes) +".csv", user_to_index)
		#Saving the like to index dictionary to a file
		#Data.write_dict_to_file(Data.outpath + "like_index_" + str(n_likes) +".csv", like_to_index)
	
	@staticmethod
	def write_dict_to_file(filepath, dict):
		w = csv.writer(open(filepath, "w"))
		for key, val in dict.items():
    			w.writerow([key, val])
	@staticmethod
	def read_dict_from_file(filepath):
		dict = {}
		for key, val in csv.reader(open(filepath)):
    			dict[int(key)] = val
		return dict

	@staticmethod
	def to_standardized(df,cols):
		for col in cols:
			##print df[col]
			##for i in df[col]:
			##	print stats.percentileofscore(list(data[col]), i)  	
			#to_add = [stats.percentileofscore(list(data[col]), i) for i in list(df[col])]
			df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
		return df	

	@staticmethod
	def aggregate_stats_for_users(user_list,weights):
		#print "agging"
		df_weight = pd.DataFrame(columns = ['userid','weight'])
		df_weight['userid'] = user_list
		df_weight['weight'] = weights
		#print df_weight.describe()
		if Data.user_personality_df is None:
			Data.user_personality_df = Data.read_df('big5.csv')
		#if Data.user_address_df is None:
		#	Data.user_address_df = Data.read_df('address.csv')	
		if Data.user_demog_df is None:
			Data.user_demog_df = Data.read_df('demog.csv')
		#print 'starting'	
		dict = {}
		##pers_means = Data.user_personality_df[Data.user_personality_df['userid'].isin(user_list)].mean()
		
		#print df_weight.head()	
		pers_means = pd.DataFrame.merge(df_weight,Data.user_personality_df,on=['userid']) 	
		#print pers_means.describe()
		#print pers_means.head()
		#print 'started percent'
		##pers_means = Data.user_personality_df[Data.user_personality_df['userid'].isin(user_list)]
		pers_means = Data.to_standardized(pers_means,['con','ext','neu','ope','agr'])
		#print 'ended percent'
		#print pers_means.head()
		for col in ['con','ext','neu','ope','agr']:
			##print pers_means.columns
			##print pers_means['weight']
			##print pers_means[col]
			pers_means[col] = pers_means[col] * pers_means['weight']/sum(list(pers_means['weight']))
			#pers_means[col] = pers_means[col]*pers_means['weight']
			dict[col] = pers_means.sum()[col]
			#print dict[col]
		 
		#temp = Data.user_address_df[Data.user_address_df['userid'].isin(user_list)].groupby(by = 'current_location_state').size()
		#temp = pd.DataFrame(temp)
		#for key, val in temp.iterrows():
		#	dict[key] = val[0]
	
		demog = pd.DataFrame.merge(Data.user_demog_df[Data.user_demog_df['userid'].isin(user_list)], df_weight, on=['userid'])
		demog = Data.to_standardized(demog, ['age', 'gender'])
		demog['age'] = demog['age']*demog['weight']/sum(list(demog['weight']))
		dict['age'] = demog.sum()['age']
		demog['gender'] = demog['gender']*demog['weight']/sum(list(demog['weight']))
		dict['gender'] = demog.sum()['gender']
		#for val in dict.keys():
		#	print '----'
		#	print val
		#	print dict[val]
		#	count = 0	
		return dict

##print Data.aggregate_stats_for_users(['99fbf129582ee8fbd0b1c5fe2d0aaabf','965e48b4a4408b08b01adfbf9dc1571b'],[.2,.3])
#Data.get_data_for_basic_model(True, 10)
#Data.get_data_for_basic_model(True, 20)
Data.get_data_for_basic_model(False, 200)

