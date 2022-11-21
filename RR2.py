import pandas as pd
from scipy import spatial
import sent2vec
from sent2vec.constants import PRETRAINED_VECTORS_PATH_WIKI, ROOT_DIR
from sent2vec.vectorizer import Vectorizer, BertVectorizer
from sent2vec.splitter import Splitter
import re
import json
import numpy

vectorizer			=	Vectorizer()

json_path			=	'RR_mean_vectors768.json'

dev_path			=	'RR_dev.csv'

chunksize			=	15
chunk_count_limit		=	5

json_file			=	open(json_path,'rt',encoding='UTF-8')
vector_means			=	json.load(json_file)
json_file.close()

for i in vector_means:
	# vector_means[i]['mean_vector']	=	numpy.array(vector_means[i]['mean_vector'])
	# vector_means[i]['mean_vector']	=	numpy.exp(vector_means[i]['mean_vector'])
	vector_means[i]['mean_vector']	=	numpy.array(vector_means[i]['mean_vector'])
	vector_means[i]['mean_vector']	=	numpy.exp(vector_means[i]['mean_vector'])
	vector_means[i]['mean_vector']	-=	vector_means[i]['positive_sizing']

df_test		=	pd.read_csv(dev_path,sep='\t',chunksize=chunksize)
predictions	=	[]

y_values	=	[]

i_count	=	1
for chunk in df_test:
	print(f'chunk {i_count}'
		,end='\r'
	)
	vectorizer.vectors	=	[]
	y_values		+=	chunk['annotation_label'].tolist()
	vectorizer.run(chunk['annotation_text'].tolist())
	for i in vectorizer.vectors:
		min_distance	=	None
		new_prediction	=	None
		for j in vector_means:
			local_cos	=	spatial.distance.cosine(vector_means[j]['mean_vector'],i)
			if min_distance == None or local_cos < min_distance:
				min_distance	=	local_cos
				new_prediction	=	j
		predictions.append(new_prediction)
	i_count	+=	1
	if chunk_count_limit != None and i_count >= chunk_count_limit:
		print(f'Reached chunk_count_limit ({chunk_count_limit})')
		break

print(len(predictions),len(y_values))
print(len([1 for i in range(len(predictions)) if predictions[i] == y_values[i]]) / len(predictions))

