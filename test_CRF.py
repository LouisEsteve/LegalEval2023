import time
import re
# import spacy
# import xml.etree.ElementTree as ET
from os import listdir
from os.path import exists
import pandas as pd
import sklearn_crfsuite
# from sklearn.metrics import precision_recall_fscore_support as PRFS
# from sklearn.metrics import flat_classification_report as FCR
# from sklearn.metrics import classification_report
# from sklearn.metrics import label_ranking_average_precision_score
# from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score

########################################

sep			=	'\t'
encoding		=	'UTF-8'

text_base_path		=	'CUSTOM_NER_TRAIN_TEXT_BASE.csv'
annot_preamble_path	=	'data/NER_TRAIN_PREAMBLE.csv'
annot_judgement_path	=	'data/NER_TRAIN_JUDGEMENT.csv'

spacy_model		=	'en_core_web_sm'

relevant_features	=	[
					'text'		,
					'__len__'	,
					'pos_'		,
					'tag_'		,
					'cluster'	,
					'is_alpha'	,
					'is_punct'	,
					'shape_'	,
					'ent_type'
				]

look_behind		=	3
look_ahead		=	3

features_memory_path	=	'features_memory.csv'
load_features_memory	=	True
# load_features_memory	=	False
save_features_memory	=	True

# https://readthedocs.org/projects/sklearn-crfsuite/downloads/pdf/latest/

########################################

def main() -> int:
	text_base_df	=	pd.read_csv(text_base_path, encoding=encoding, sep=sep, index_col=0
	# , nrows=3000
	)
	preamble_df	=	pd.read_csv(annot_preamble_path, encoding=encoding, sep=sep)
	judgement_df	=	pd.read_csv(annot_judgement_path, encoding=encoding, sep=sep)
	
	annot_df	=	pd.concat((preamble_df,judgement_df))
	# print(annot_df)
	
	X_features	=	[]
	y_values	=	[]
	
	if load_features_memory and exists(features_memory_path):
		features_df	=	pd.read_csv(features_memory_path,encoding=encoding,sep=sep)
		X_features	=	features_df.to_dict('records')
		
		for i in X_features:
			# del(i['y_value'])
			i.pop('y_value')
		
		
		y_values	=	features_df['y_value'].to_list()
		print(f'Loaded {features_memory_path}')
	else:
		
		nlp		=	spacy.load(spacy_model)
		
		
		# for i in text_base_df['id'][:25]:
		for i in text_base_df['id']:
			print(i,end='\r')
			# doc		=	nlp(text_base_df.loc[text_base_df['id'] == i]['text'])
			# doc		=	nlp(text_base_df['text'][text_base_df['id'] == i])
			doc		=	nlp(text_base_df.loc[(text_base_df['id'] == i)]['text'].to_list()[0])
			for j in doc:
				features	=	{}
				for k in relevant_features:
					attr			=	getattr(j,k)
					attr			=	attr if '__call__' not in dir(attr) else attr()
					# features[k]		=	attr
					features[k.lstrip('_')]	=	attr
				X_features.append(features)
				# print(dir(j))
				# print(features)
				# print(j.left_edge,j.right_edge)
				# return 2
				
				# tok_start		=	j.i
				# tok_end			=	j.i + len(j)
				
				tok_start		=	j.idx
				tok_end			=	j.idx + len(j)
				relevant_annotations	=	annot_df.loc[(annot_df['text_id'] == i) & (annot_df['annotation_start'] <= tok_start) & (annot_df['annotation_end'] >= tok_end)]
				if len(relevant_annotations) > 0:
					y_values.append(relevant_annotations['annotation_label'].to_list()[0])
				else:
					y_values.append('NONE')
				# if type(y_values[-1]) != 'str' or len(y_values[-1]) == 0:
					# y_values[-1] = 'NONE'
		if save_features_memory:
			for i in range(len(y_values)):
				X_features[i]['y_value']	=	y_values[i]
			memory_df	=	pd.DataFrame.from_dict(X_features)
			memory_df.to_csv(features_memory_path,encoding=encoding,sep=sep)
			print(f'Savec {features_memory_path}')
	
	"""
	X_features_copy	=	X_features[:]
	for i in range(len(X_features)):
		if look_behind > 0:
			j_count	=	-look_behind
			for j in range(max(0,i-look_behind), i):
				# new_key	=	f'{j_count}'
				for k in X_features_copy[j]:
					new_key	=	f'{j_count}:{k}'
					X_features[i][new_key]	=	X_features_copy[j][k]
				j_count	+=	1

		if look_ahead > 0:
			j_count	=	1
			for j in range(i+1, min(len(X_features),i+look_ahead)):
				# new_key	=	f'{j_count}'
				for k in X_features_copy[j]:
					new_key	=	f'+{j_count}:{k}'
					X_features[i][new_key]	=	X_features_copy[j][k]
				j_count	+=	1
	"""

	len_X_features	=	len(X_features)
	# for i in X_features:
	for i in range(len_X_features):
		for j in range(-look_behind,look_ahead+1):
			if j == 0:
				continue
			if (i+j) < 0 or (i+j) >= len_X_features:
				continue
			local_index	=	i+j
			for k in X_features[local_index]:
				if k.startswith('-') or k.startswith('+'):
					continue
				k_value			=	X_features[local_index][k]
				new_key			=	f'{j}:{k}' if j < 0 else f'+{j}:{k}'
				# print(new_key)
				# X_features[i][k]	=	k_value
				X_features[i][new_key]	=	k_value
				
	for i in X_features:
		if 'bias' not in i:
			i['bias']	=	1.0
			
	print(X_features[3])
	
	assert len(X_features) == len(y_values)
	# print(len(X_features),len(y_values))
	# print(X_features,y_values)
	
	# for i in range(len(X_features)-1):
		# assert X_features[i].keys() == X_features[i+1].keys()
	
	
	
	X_features	=	[X_features]
	y_values	=	[y_values]
	
	CRF_model	=	sklearn_crfsuite.CRF(
		# algorithm='lbfgs',
		# algorithm='l2sgd',
		algorithm='arow',
		# algorithm='pa',
		# c1=0.1,
		# c2=0.1,
		max_iterations=150,
		all_possible_transitions=True,
		verbose=True
	)
	
	print('\nTraining...')
	CRF_model.fit(X_features, y_values)
	
	prediction	=	CRF_model.predict(X_features)
	
	# print(prediction)
	# print(y_values)
	
	# local_PRFS	=	PRFS(prediction,y_values)
	# print(local_PRFS)
	# print(FCR(prediction,y_values))
	# print(classification_report(prediction,y_values))
	# print(label_ranking_average_precision_score(prediction,y_values))
	# print(multilabel_confusion_matrix(prediction,y_values))
	# print(accuracy_score(prediction,y_values))
	
	for i in range(len(prediction)):
		print(i,':')
		print(accuracy_score(prediction[i],y_values[i]))
		top_count	=	0
		bot_count	=	0
		for j in range(len(prediction[i])):
			pred_value	=	prediction[i][j]
			actu_value	=	y_values[i][j]
			if actu_value != 'NONE':
				bot_count	+=	1
				if pred_value == actu_value:
					top_count	+=	1
		print(top_count/bot_count)
	
	return 0

########################################

if __name__ == '__main__':
	t		=	time.time()
	main_result	=	main()
	print(f'Program ended with result {main_result} in {time.time()-t}s')