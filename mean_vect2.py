import os
from scipy import spatial
import sent2vec
import sys
from sent2vec.constants import PRETRAINED_VECTORS_PATH_WIKI, ROOT_DIR
from sent2vec.vectorizer import Vectorizer, BertVectorizer
from sent2vec.splitter import Splitter
import pandas as pd
import numpy
import re
import json

vectorizer = Vectorizer()

chunksize			=	15

df_train			=	pd.read_csv('RR_TRAIN.csv',sep='\t',chunksize=chunksize)
# df_train['annotation_text']	=	df_train['annotation_text'].str.strip(' \t\n')

chunk_count_limit		=	25


# RR = ['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE']
# RR			=	df_train['annotation_label'].unique()

"""
split_pattern 		= '(?<=\.\?!)\\s*([A-Z])'
split_pattern_2 	= '(?<=[ˆA-Z]{2}[\.\?!])\s+(?=[A-Z])'

regex_splitter		=	re.compile(split_pattern_2)
"""


BERT_MODEL_SIZE		=	768
f_out_path		=	'RR_mean_vectors768.json'

# output_obj	=	{'vectors_list':[]}
output_obj	=	{}

# CALCUL MOYENNE / CLASSE

i_count	=	1
for chunk in df_train:
	print(f'chunk {i_count}',end='\r')
	chunk['annotation_text']	=	chunk['annotation_text'].str.strip(' \t\n')

	# for tag in RR:
	for tag in chunk['annotation_label'].unique():
		# local_df			=	df_train[df_train['annotation_label'] == tag]
		local_df			=	chunk[chunk['annotation_label'] == tag]
		len_local_df			=	len(local_df)
		# if len_local_df == 0:
			# continue
		
		sentences			=	local_df['annotation_text'].to_list()
		
		vectorizer.run(sentences)
		vectors				=	vectorizer.vectors
		mean_vector			=	numpy.mean(vectors,axis=0)
		# output_obj['vectors_list'].append({'tag':tag,'vector_size':BERT_MODEL_SIZE,'units_count':len(vectors),'mean_vector':mean_vector.tolist()})
		if tag not in output_obj:
			output_obj[tag]	=	{'vector_size':BERT_MODEL_SIZE,'units_count':0,'mean_vector':numpy.zeros(BERT_MODEL_SIZE)}
		"""
		previous_sum			=	output_obj[tag]['mean_vector'] * output_obj[tag]['units_count']
		sum				=	previous_sum + mean_vector * len_local_df
		output_obj[tag]['units_count']	+=	len_local_df 
		new_mean_vector			=	sum / len_local_df
		output_obj[tag]['mean_vector']	=	new_mean_vector
		"""
		
		new_unit_count			=	output_obj[tag]['units_count'] + len_local_df
		new_mean_vector			=	output_obj[tag]['mean_vector'] * (output_obj[tag]['units_count'] / new_unit_count) + mean_vector / len_local_df
		output_obj[tag]['units_count']	+=	len_local_df 
		output_obj[tag]['mean_vector']	=	new_mean_vector
		# print(f'Added {tag}, {len(output_obj["vectors_list"])}')
	i_count	+=	1
	if chunk_count_limit != None and i_count >= chunk_count_limit:
		print(f'Reached chunk_count_limit ({chunk_count_limit})')
		break

for i in output_obj:
	output_obj[i]['mean_vector']	=	output_obj[i]['mean_vector'].tolist()


f_out	=	open(f_out_path,'wt',encoding='UTF-8')
json.dump(output_obj,f_out,indent=8)
f_out.close()

print(f'Wrote {f_out_path}')