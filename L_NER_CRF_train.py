"""
AUTHOR		:	UNIVERSITY OF ORLEANS (FRANCE)
TASK		:	LEGALEVAL2023
SUB-TASK	:	L-NER
***************************************************************
FUNCTIONS DEFINED IN THIS SCRIPT
- IOB_correcter
- IOB2_transformer
- IOBES_transformer
- seqeval_report
- prepare_model
- generate_features
- features_and_values_for_CRF
- train
- levenshtein_distance
- post_traitement
- estimate_performance_on_dev
"""

import time
import re
# import spacy
from os import listdir
from os.path import exists
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import json
import sklearn_crfsuite
# from sklearn.metrics import accuracy_score
from seqeval.metrics.sequence_labeling import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2, IOBES

########################################

config_path		=	'L_NER_CRF_default_config.json'
# config_path		=	'CRF_MULTI95_CRF3_config.json'

# https://readthedocs.org/projects/sklearn-crfsuite/downloads/pdf/latest/

########################################

config				=	None

########################################

anti_whitespace_regex		=	re.compile('\\S')

date_regex			=	re.compile('((0?\\d|[12]\\d|3[01])(?P<SEP1>\\D)(0?\\d|1[012])(?P=SEP1)(18|19|20)\\d{2})|((0?\\d|1[012])(?P<SEP2>\\D)(0?\\d|[12]\\d|3[01])(?P=SEP2)(18|19|20)\\d{2})|((18|19|20)\\d{2}(?P<SEP3>\\D)(0?\\d|[12]\\d|3[01])(?P=SEP3)(0?\\d|1[012]))|((18|19|20)\\d{2})(?P<SEP4>\\D)(0?\\d|1[012])(?P=SEP4)(0?\\d|[12]\\d|3[01])')
org_regex			=	re.compile('[A-Z]{3,}')

CRF_model			=	None
annot_df			=	None

def IOB_correcter(
		x	:	list
	):
	'''
	This function ensures that all annotations start with a B and not an I according to IOB-like formats.
	
	def IOB_correcter(
			x	:	list
		):
		[...]
	'''
	########
	for i in range(len(x)):
		previous_IOB	=	'O'
		for j in range(len(x[i])):
			if x[i][j][0] == 'I':
				if previous_IOB not in ['B','I']:
					x[i][j]	=	f'B{x[i][j][1:]}'
			previous_IOB	=	x[i][j][0]
	return x

def IOB2_transformer(
		x		:	list,
		null_class	:	str	=	'NONE'
	) -> list:
	'''
	This function transforms x into an IOB2-compliant format.
	Input and output are both a list of lists of strings.
	It expects null_class as the equivalent of 'O', which by default is set to 'NONE'.
	
	def IOB2_transformer(
			x		:	list,
			null_class	:	str	=	'NONE'
		) -> list:
		[...]
	'''
	########
	global_output	=	[]
	for i in x:
		local_output	=	[]
		previous_class	=	None
		for j in i:
			prefix	=	''
			if j != previous_class:
				if j != null_class:
					prefix	=	'B-'
			else:
				prefix	=	'I-'
			if j == null_class:
				local_output.append('O')
			else:
				local_output.append(f'{prefix}{j}')
			previous_class	=	j
		global_output.append(local_output)
	return global_output

def IOBES_transformer(
		x		:	list,
		null_class	:	str	=	'NONE'
	) -> list:
	'''
	This function transforms x into an IOBES-compliant format.
	Input and output are both a list of lists of strings.
	It expects null_class as the equivalent of 'O', which by default is set to 'NONE'.
	
	def IOBES_transformer(
			x		:	list,
			null_class	:	str	=	'NONE'
		) -> list:
		[...]
	'''
	########
	global_output	=	[]
	for i in x:
		local_output	=	[]
		previous_class	=	None
		for j in i:
			prefix	=	''
			if j != previous_class:
				if len(local_output) > 0:
					if local_output[-1][0] == 'B':
						local_output[-1]	=	f'S{local_output[-1][1:]}'
					elif local_output[-1][0] == 'I':
						local_output[-1]	=	f'E{local_output[-1][1:]}'
				
				if j != null_class:
					prefix	=	'B-'
			else:
				prefix	=	'I-'
			if j == null_class:
				local_output.append('O')
			else:
				local_output.append(f'{prefix}{j}')
			previous_class	=	j
		#####
		# END OF DOCUMENT FIX
		if len(local_output) > 0:
			if local_output[-1][0] == 'B':
				local_output[-1]	=	f'S{local_output[-1][1:]}'
			elif local_output[-1][0] == 'I':
				local_output[-1]	=	f'E{local_output[-1][1:]}'
		#####
		global_output.append(local_output)
	return global_output

def seqeval_report(
		y_values	:	list,
		prediction	:	list
	):
	'''
	This function prints a seqeval report based on the two inputs.
	
	def seqeval_report(
			y_values	:	list,
			prediction	:	list
		):
		[...]
	'''
	########
	if "early_IOB_mapping" in config:
		if not config["early_IOB_mapping"]:
			if "IOBES" in config and config["IOBES"]:
				IOBES_y_values		=	IOBES_transformer(y_values)
				IOBES_prediction	=	IOBES_transformer(prediction)
				print(
					seqeval_classification_report(
								IOBES_y_values,
								IOBES_prediction,
						scheme	=	IOBES,
						digits	=	3,
						mode	=	'strict' if 'mode' not in config else config["mode"]
					)
				)
			else:
				"""
				IOB_y_values	=	[]
				IOB_prediction	=	[]
				for i in range(len(y_values)):
					reset_IOB_mapper()
					local_y_values		=	[]
					for j in range(len(y_values[i])):
						local_y_values.append(IOB_mapper(y_values[i][j]))
					IOB_y_values.append(local_y_values)
					
					reset_IOB_mapper()
					local_prediction	=	[]
					for j in range(len(y_values[i])):
						local_prediction.append(IOB_mapper(prediction[i][j]))
					IOB_prediction.append(local_prediction)
				"""
				IOB_y_values	=	IOB2_transformer(y_values)
				IOB_prediction	=	IOB2_transformer(prediction)
				print(
					seqeval_classification_report(
								IOB_y_values,
								IOB_prediction,
						scheme	=	IOB2,
						digits	=	3,
						mode	=	'strict' if 'mode' not in config else config["mode"]
					)
				)
		else:
			if "IOBES" in config and config["IOBES"]:
				print(
					seqeval_classification_report(
								y_values,
								prediction,
						scheme	=	IOBES,
						digits	=	3,
						mode	=	'strict' if 'mode' not in config else config["mode"]
					)
				)
			else:
				prediction	=	IOB_correcter(prediction)
				print(
					seqeval_classification_report(
								y_values,
								prediction,
						scheme	=	IOB2,
						digits	=	3,
						mode	=	'strict' if 'mode' not in config else config["mode"]
					)
				)
	
	return y_values, prediction

def prepare_model():
	'''
	This function prepares the model which is then stored in CRF_model (global).
	It bases itself in the config object that comes from the JSON configuration file.
	
	def prepare_model():
		[...]
	'''
	########
	global CRF_model
	
	if CRF_model == None and config["load_model"] and exists(config["model_path"]):
		try:
			CRF_model	=	joblib.load(config["model_path"])
			print(f'Loaded {config["model_path"]}')
		except OSError or IOError as e:
			print(e)
			return 1
	else:
		CRF_model	=	sklearn_crfsuite.CRF(
			# model_filename			=	config["model_path"],
			algorithm			=	config["solver"],
			c1				=	None if config["solver"].lower()!='lbfgs' else config["c1"],
			c2				=	None if config["solver"].lower() not in ['lbfgs','l2sgd'] else config["c2"],
			pa_type				=	None if config["solver"].lower()!='pa' or "pa_type" not in config else config["pa_type"],
			max_iterations			=	config["max_epoch"],
			gamma				=	None if config["solver"].lower()!='arow' else config["gamma"],
			all_possible_transitions	=	config["all_possible_transitions"],
			all_possible_states		=	config["all_possible_states"],
			variance			=	None if config["solver"].lower()!='arow' else config["variance"],
			epsilon				=	None if config["solver"].lower()=='l2sgd' else config["epsilon"],
			period				=	None if config["solver"].lower() not in ['l2sgd','lbfgs'] or "period" not in config else config["period"],
			verbose				=	True,
			min_freq			=	0.0 if "min_freq" not in config else config["min_freq"],
			# calibration_eta			=	config["calibration_eta"] if "calibration_eta" in config else 0.1,
			# calibration_rate		=	config["calibration_rate"] if "calibration_rate" in config else 2.0,
			# calibration_samples		=	config["calibration_samples"] if "calibration_samples" in config else 1000,
			# calibration_candidates		=	config["calibration_candidates"] if "calibration_candidates" in config else 10,
			# calibration_max_trials		=	config["calibration_max_trials"] if "calibration_max_trials" in config else 20
		)
	return 0

def generate_features(
		text_base_path		:	str,
		features_memory_path	:	str,
		annot_preamble_path	:	str,
		annot_judgement_path	:	str
	):
	'''
	This function is to be called if no features were stored previously.
	It generates the features using SpaCy based on the config parameters and saves them accordingly for them to be reloaded later.
	
	def generate_features(
			text_base_path		:	str,
			features_memory_path	:	str,
			annot_preamble_path	:	str,
			annot_judgement_path	:	str
		):
		[...]
	'''
	########
	preamble_df	=	pd.read_csv(
					annot_preamble_path,
		encoding	=	config["encoding"],
		sep		=	config["sep"]
	)
	preamble_df.dropna(how='all',inplace=True)
	
	judgement_df	=	pd.read_csv(
					annot_judgement_path,
		encoding	=	config["encoding"],
		sep		=	config["sep"]
	)
	judgement_df.dropna(how='all',inplace=True)
	
	annot_df	=	pd.concat((preamble_df,judgement_df))
	
	del(preamble_df)
	del(judgement_df)
	
	
	X_features	=	[]
	y_values	=	[]
	
	text_base_df		=	pd.read_csv(text_base_path, encoding=config["encoding"], sep=config["sep"])
	text_base_df.dropna(how='all',inplace=True)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\n','\n',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\t','\t',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\r','\r',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\"','"',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace("\\'","'",regex=False)
	
	text_base_df.sort_values(['id'],kind='mergesort',inplace=True)
	
	if 'spacy' not in dir():
		import spacy
	nlp		=	spacy.load(config["spacy_model"])
	print(f'Loaded {config["spacy_model"]}')
	
	id_list	=	text_base_df['id'].to_list()
	
	i_count	=	1
	i_limit	=	len(text_base_df)
	for doc in nlp.pipe(text_base_df['text'].to_list(),batch_size=32):
		i	=	id_list[i_count-1]
		print(f'Generating features for text {i_count:>6} / {i_limit} : {i}',end='\r')
		# REWORK HERE
		for j in doc:
			if anti_whitespace_regex.search(j.text) == None:
				continue
			features	=	{
				'text_id'	:	i,
				'id_in_text'	:	j.i
			}
			for k in config["relevant_features"]:
				attr			=	getattr(j,k)
				attr			=	attr if '__call__' not in dir(attr) else attr()
				features[k.lstrip('_')]	=	attr if type(attr) in [bool,int,str] else str(attr)
			X_features.append(features)
			
			tok_start			=	j.idx
			tok_end				=	j.idx + len(j)
			features['offset_start']	=	tok_start
			features['offset_end']		=	tok_end
			relevant_annotations		=	annot_df.loc[(annot_df['text_id'] == i) & (annot_df['annotation_start'] <= tok_start) & (annot_df['annotation_end'] >= tok_end)]
			if len(relevant_annotations) > 0:
				y_values.append(relevant_annotations['annotation_label'].to_list()[0])
			else:
				y_values.append('NONE')
		i_count	+=	1
	print('') # TO HAVE A LINEFEED
	
	# SAVING
	for i in range(len(y_values)):
		X_features[i]['y_value']	=	y_values[i]
	memory_df		=	pd.DataFrame.from_dict(X_features)
	memory_df.index.name	=	'id'
	memory_df.to_csv(features_memory_path,encoding=config["encoding"],sep=config["sep"])
	print(f'Saved {features_memory_path}')
	return memory_df

def features_and_values_for_CRF(
		text_base_path		:	str,
		features_memory_path	:	str,
		annot_preamble_path	:	str,
		annot_judgement_path	:	str,
		clean			:	bool	=	False,
		clean_min_count		:	int	=	0
	):
	'''
	This function returns features, prediction values, and meta information, each in a separate list.
	Therefore, results are to be obtained through :
		features, values, meta	=	features_and_values_for_CRF(...)
	Features and predictions values are the most important, meta information are mostly given for postprocessing and potentially future changes.
	
	
	def features_and_values_for_CRF(
			text_base_path		:	str,
			features_memory_path	:	str,
			annot_preamble_path	:	str,
			annot_judgement_path	:	str,
			clean			:	bool	=	False,
			clean_min_count		:	int	=	0
		):
		[...]
	'''
	########
	X_features	=	[]
	y_values	=	[]
	meta_info	=	[]
	
	if not exists(features_memory_path):
		print('Features were not found, generating them...')
		features_df	=	generate_features(
			text_base_path,
			features_memory_path,
			annot_preamble_path,
			annot_judgement_path
		)
	else:
		features_df	=	pd.read_csv(
						features_memory_path,
			encoding	=	config["encoding"],
			sep		=	config["sep"],
			index_col	=	'id',
			na_values	=	'N/A',
			dtype		=	config["dtype"]
		)
		print(f'Loaded features from {features_memory_path}')
	features_df.sort_values(['text_id','id_in_text'],kind='mergesort',inplace=True)
	
	if "drop_rows_if_true" in config:
		len_df	=	len(features_df)
		
		for i in config["drop_rows_if_true"]:
			if i in features_df:
				features_df	=	features_df[features_df[i] != True]
		
		print(f'Dropped {len_df-len(features_df)}')
		
	meta_info	=	features_df[['text_id','id_in_text','offset_start','offset_end']].to_dict('records')
	features_df	=	features_df[[i.lstrip('_') for i in config["relevant_features"] if i not in config["irrelevant_features"]] + ['y_value','text_id']]
	
	##################
	# CLEANING TRAIN
	
	if clean:
		removed_count		=	0
		acceptable_texts	=	[]
		features_df_gb		=	features_df.groupby('text_id')
		for name, group in features_df_gb:
			if group['y_value'].unique().tolist() == ['NONE'] and len(group) > clean_min_count:
				removed_count	+=	1
			else:
				acceptable_texts.append(name)
		acceptable_texts	=	pd.Series(acceptable_texts)
		acceptable_texts.name	=	'text_id'
		features_df		=	pd.merge(features_df, acceptable_texts)
		print(f'Removed {removed_count} texts in cleaning')
	
	
	##################
	"""
	if "drop_rows_if_true" in config:
		len_df	=	len(features_df)
		
		for i in config["drop_rows_if_true"]:
			features_df	=	features_df[features_df[i] != True]
		
		print(f'Dropped {len_df-len(features_df)}')
	"""
	##################
	
	features_df.dropna(how='all',inplace=True)
	
	for i in config["labels_to_erase"]:
		features_df['y_value']	=	features_df['y_value'].str.replace(i,'NONE',regex=False)
	
	###################################
	
	all_text_ids		=	features_df['text_id'].unique()#[:50]
	
	columns_to_consider	=	[i for i in features_df.columns if i not in ['y_value','text_id','offset_start','offset_end']]
	
	features_df_gb		=	features_df.groupby('text_id')
	
	for j in range(-config["look_behind"],config["look_ahead"]+1):
		if j == 0:
			continue
		prefix	=	f'{"" if j < 0 else "+"}{j}:'
		for k in columns_to_consider:
			new_column_name	=	f'{prefix}{k}'
			features_df[new_column_name]	=	features_df_gb[k].shift(-j)
	
	features_df_gb	=	features_df.groupby('text_id')
	i_count	=	1
	i_limit	=	len(all_text_ids)
	for name, group in features_df_gb:
		# if i_count % 100 == 0 or i_count == i_limit:
		print(f'y_values : {i_count:>6} / {i_limit}',end='\r')
		y_values.append(group['y_value'].to_list())
		i_count	+=	1
	
	if "early_IOB_mapping" in config and config["early_IOB_mapping"]:
		if "IOBES" in config and config["IOBES"]:
			y_values	=	IOBES_transformer(y_values)
		else:
			"""
			for i in range(len(y_values)):
				reset_IOB_mapper()
				for j in range(len(y_values[i])):
					y_values[i][j]	=	IOB_mapper(y_values[i][j])
			"""
			y_values	=	IOB2_transformer(y_values)
	
	features_df.pop('y_value')
	features_df.pop('text_id')
	i_count	=	1
	for name, group in features_df_gb:
		print(f'X_features : {i_count:>6} / {i_limit}',end='\r')
		X_features.append(group.to_dict('records'))
		i_count	+=	1
	
	for i in X_features:
		###
		first_word	=	i[0]['text'] if 'text' in i[0] else None
		last_word	=	i[-1]['text'] if 'text' in i[-1] else None
		text_root	=	None
		for j in i:
			try:
				if ('head' in j and 'text' in j) and j['head'] == j['text']:
					# text_root	=	getattr(j,'head')
					text_root	=	j['head']
					break
			except:
				pass
		###
		for j in i:
			keys	=	list(j.keys())
			for k in keys:
				if j[k] == None or type(j[k]) not in [bool,int,str]:
					j.pop(k)
					###
					
					if first_word:
						j['first_word']	=	first_word
					if last_word:
						j['last_word']	=	last_word
					if text_root:
						j['text_root']	=	text_root
					
					###
	del(features_df)
	return X_features, y_values, meta_info
	
def train():
	'''
	This function trains the model stored in CRF_model.
	
	def train():
		[...]
	'''
	########
	global CRF_model
	if CRF_model == None:
		prepare_model()
	X_features, y_values, meta_info	=	features_and_values_for_CRF(
					config["text_base_train_path"],
					config["features_memory_train_path"],
					config["annot_preamble_train_path"] if "annot_preamble_train_path" in config else 'NONE',
					config["annot_judgement_train_path"] if "annot_judgement_train_path" in config else 'NONE',
		clean		=	config["clean_TRAIN"] if "clean_TRAIN" in config else False,
		clean_min_count	=	config["clean_min_count"] if "clean_min_count" in config else 0
	)
	
	del(meta_info)
	
	CRF_model.fit(X_features,y_values)
	
	################
	"""
	prediction		=	CRF_model.predict(X_features)
	
	print("Performance on TRAIN:")
	y_values, prediction	=	seqeval_report(
		y_values,
		prediction
	)
	"""
	################
	
	
	if config["save_model"]:
		try:
			joblib.dump(CRF_model, config["model_path"])
		except IOError or OSError as e:
			print(e)
			print(f'WARNING: COULD NOT SAVE MODEL TO PATH {config["model_path"]}, SEE MESSAGE JUST ABOVE')
		else:
			print(f'Saved {config["model_path"]}')
	

def levenshtein_distance(
		a	:	str,
		b	:	str
	) -> int:
	'''
	This function computes the Levenshtein distance between two strings.
	
	def levenshtein_distance(
			a	:	str,
			b	:	str
		) -> int:
		[...]
	'''
	########
	len_a	=	len(a)
	len_b	=	len(b)
	if len_a == 0:
		return len_b
	if len_b == 0:
		return len_a
	if a[0] == b[0]:
		return levenshtein_distance(a[1:],b[1:])
	c	=	levenshtein_distance(a[1:],b[1:])
	d	=	levenshtein_distance(a,b[1:])
	e	=	levenshtein_distance(a[1:],b)
	return 1 + min(c,d,e)

def post_traitement(
		X_features		:	list,
		prediction		:	list,
		enable_cities_query	:	bool	=	False,
		enable_DATE_regex	:	bool	=	False,
		enable_ORG_regex	:	bool	=	False
	) -> list:
	'''
	This function takes the features and predictions, and returns the predictions once post-processing has been done.
	
	def post_traitement(
			X_features	:	list,
			prediction	:	list
		) -> list:
	'''
	#<POST_TRAITEMENT>
	
	# if "enable_cities_query" in config and config["enable_cities_query"]:
	if enable_cities_query:
		try:
			cities_df		=	pd.read_csv('cities.csv' if "cities_path" not in config else config["cities_path"],sep='\t',encoding='UTF-8')
		except OSError or IOError as e:
			print(e)
			print("Could not open path to cities file. See message above.")
		else:
			cities_df['city']	=	cities_df['city'].str.lower()
			cities_df['state']	=	cities_df['state'].str.lower()
			cities_and_states		=	cities_df['city'].tolist() + cities_df['state'].tolist()
			for i in range(len(X_features)):
				max_j	=	len(X_features[i])
				for j in range(max_j):
					if prediction[i][j] != 'O':
						continue
					if 'text' not in X_features[i][j] or len(X_features[i][j]['text']) < 6:
						continue
					if j == 0 or 'text' not in X_features[i][j-1] or X_features[i][j-1]['text'].lower() != X_features[i][j-1]:
						continue
					local_lower	=	X_features[i][j]['text'].lower()
					for k in cities_and_states:
						if local_lower in k:
							prediction[i][j]	=	'I-GPE'
							break
	prediction	=	IOB_correcter(prediction)
	
	
	
	# if "enable_DATE_regex" in config and config["enable_DATE_regex"]:
	if enable_DATE_regex:
		for i in range(len(X_features)):
			max_j	=	len(X_features[i])
			# for j in range(len(X_features[i])):
			for j in range(max_j):
				if 'text' not in X_features[i][j]:
					continue
				if date_regex.fullmatch(X_features[i][j]['text'].strip()):
					prediction[i][j]	=	'B-DATE'
	
	# if "enable_ORG_regex" in config and config["enable_ORG_regex"]:
	if enable_ORG_regex:
		for i in range(len(X_features)):
			max_j	=	len(X_features[i])
			# for j in range(len(X_features[i])):
			for j in range(max_j):
				if 'text' not in X_features[i][j]:
					continue
				if prediction[i][j] != 'O':
					continue
				if org_regex.fullmatch(X_features[i][j]['text'].strip()):
					prediction[i][j]	=	'B-ORG'
	
	'''
	# low
	PERSON_types	=	['JUDGE','LAWYER','RESPONDENT','PETITIONER','WITNESS','OTHER_PERSON']
	person_dict	=	{}
	for i in range(len(X_features)):
		current_PERSON	=	''
		start_index	=	None
		for j in range(len(X_features[i])):
			# current_class	=	prediction[i][j][2:]
			# if current_class in PERSON_types:
			if prediction[i][j][2:] in PERSON_types:
				current_class	=	prediction[i][j][2:]
				if current_PERSON == '':
					start_index	=	j
				else:
					current_PERSON	=	f'{current_PERSON} '
				current_PERSON	=	f'{current_PERSON}{X_features[i][j]["text"]}'
			else:
				if current_PERSON != '':
					"""
					min_lev_distance	=	1000
					min_index		=	None
					len_current_PERSON	=	len(current_PERSON)
					for k in person_dict:
						if abs(len_current_PERSON-len(k)) > 2 or len_current_PERSON < 32 or len(k) < 32:
							continue
						local_lev_distance	=	levenshtein_distance(k,current_PERSON)
						if local_lev_distance < min_lev_distance:
							min_lev_distance	=	local_lev_distance
							min_index		=	k
					if min_index != None and min_lev_distance < 5:
						current_PERSON	=	min_index
					"""
					
					if current_PERSON not in person_dict:
						# person_dict[current_PERSON]	=	{'count':0,'instances':[]}
						person_dict[current_PERSON]	=	{'count':0,'count_per_class':{},'instances':[]}
					person_dict[current_PERSON]['count']	+=	1
					if current_class not in person_dict[current_PERSON]['count_per_class']:
						person_dict[current_PERSON]['count_per_class'][current_class]	=	0
					person_dict[current_PERSON]['count_per_class'][current_class]	+=	1
					person_dict[current_PERSON]['instances'].append((i,start_index))
					current_PERSON	=	''
	'''
	
	
	"""
	#DETRIMENTAL
	for i in person_dict:
		if person_dict[i]['count'] == 1:
			# for j in person_dict[i]['instances']:
			for j in range(len(person_dict[i]['instances'])):
				# prediction[i][j]	=	'B-OTHER_PERSON'
				# print(person_dict[i]['instances'][j])
				prediction[person_dict[i]['instances'][j][0]][person_dict[i]['instances'][j][1]]	=	'B-OTHER_PERSON'
				# k	=	j+1
				k	=	person_dict[i]['instances'][j][1]+1
				# k_limit	=	len(prediction[i])
				k_limit	=	len(prediction[person_dict[i]['instances'][j][0]])
				while k < k_limit:
					# if prediction[i][k][0] == 'I':
					if prediction[person_dict[i]['instances'][j][0]][k][0] == 'I':
						# prediction[i][k]	=	'I-OTHER_PERSON'
						prediction[person_dict[i]['instances'][j][0]][k]	=	'I-OTHER_PERSON'
						k	+=	1
					else:
						break
	"""
	
	"""
	for i in person_dict:
		new_class	=	max(person_dict[i]['count_per_class'])
		# print(i,person_dict[i]['count_per_class'],new_class)
		for j in range(len(person_dict[i]['instances'])):
			# print(person_dict[i]['instances'][j])
			prediction[person_dict[i]['instances'][j][0]][person_dict[i]['instances'][j][1]]	=	f'B-{new_class}'
			k	=	person_dict[i]['instances'][j][1]+1
			k_limit	=	len(prediction[person_dict[i]['instances'][j][0]])
			while k < k_limit:
				if prediction[person_dict[i]['instances'][j][0]][k][0] == 'I':
					prediction[person_dict[i]['instances'][j][0]][k]	=	f'I-{new_class}'
					k	+=	1
				else:
					break
	"""
	
	return prediction
	


def estimate_performance_on_dev():
	'''
	This function estimates the performance on the DEV set using the model stored in CRF_model.
	
	def estimate_performance_on_dev():
		[...]
	'''
	########
	global CRF_model
	if CRF_model == None:
		print('No CRF_model available, cannot estimate performance on DEV')
		exit()
	X_features, y_values, meta_info	=	features_and_values_for_CRF(config["text_base_dev_path"], config["features_memory_dev_path"], config["annot_preamble_dev_path"] if "annot_preamble_dev_path" in config else 'NONE', config["annot_judgement_dev_path"] if "annot_judgement_dev_path" in config else 'NONE')
	
	prediction		=	CRF_model.predict(X_features)
	
	print("Performance on DEV before all post-processings:")
	y_values, prediction	=	seqeval_report(
		y_values,
		prediction
	)
	
	#<POST_TRAITEMENT>
	
	prediction	=	post_traitement(
		X_features		=	X_features,
		prediction		=	prediction,
		enable_cities_query	=	"enable_cities_query" in config and config["enable_cities_query"],
		enable_DATE_regex	=	"enable_DATE_regex" in config and config["enable_DATE_regex"],
		enable_ORG_regex	=	"enable_ORG_regex" in config and config["enable_ORG_regex"]
	)
	
	print("Performance on DEV after all post-processings:")
	if "IOBES" in config and config["IOBES"]:
		print(seqeval_classification_report(y_values,prediction,scheme=IOBES,digits=3,mode='strict' if 'mode' not in config else config["mode"]))
	else:
		prediction	=	IOB_correcter(prediction)
		print(seqeval_classification_report(y_values,prediction,scheme=IOB2,digits=3,mode='strict' if 'mode' not in config else config["mode"]))
	#</POST_TRAITEMENT>
	
	flat_prediction	=	[]
	flat_y_values	=	[]
	for i in range(len(prediction)):
		flat_prediction	+=	prediction[i]
		flat_y_values	+=	y_values[i]
	
	X_text_col	=	[]
	y_value_col	=	[]
	y_predict_col	=	[]
	
	for i in range(len(X_features)):
		for j in range(len(X_features[i])):
			# X_text_id.append(X_features[i][j][])
			if 'text' in X_features[i][j]:
				X_text_col.append(X_features[i][j]['text'])
			else:
				X_text_col.append(' ')
			y_value_col.append(y_values[i][j])
			y_predict_col.append(prediction[i][j])
	
	results_df			=	pd.DataFrame()
	results_df['text_id']		=	pd.Series([i['text_id'] for i in meta_info],dtype='string')
	results_df['id_in_text']	=	pd.Series([i['id_in_text'] for i in meta_info],dtype='Int32')
	results_df['offset_start']	=	pd.Series([i['offset_start'] for i in meta_info],dtype='Int32')
	results_df['offset_end']	=	pd.Series([i['offset_end'] for i in meta_info],dtype='Int32')
	results_df['y_value']		=	pd.Series(y_value_col,dtype='category')
	results_df['y_predict']		=	pd.Series(y_predict_col,dtype='category')
	results_df['X_text']		=	pd.Series(X_text_col,dtype='string')
	results_df.index.name		=	'id'
	results_df.to_csv('latest_results.csv',sep=config["sep"],encoding=config["encoding"])
	print(f'Wrote latest_results.csv')
	
	"""
	# THIS BLOCK OF CODE WAS FOR WHEN WE DID TOKEN-PER-TOKEN SCORING; NOT USEFUL ANYMORE CONSIDERING WE USE SEQEVAL
	
	iob_counts		=	{}
	type_counts		=	{}
	
	
	class_dict	=	{'NONE':{'TP':0,'FP':0,'FN':0}}
	top_count	=	0
	bot_count	=	0
	
	for i in range(len(prediction)):
		previous_actual		=	'NONE'
		previous_predicted	=	'NONE'
		
		overlap_offsets_actual		=	[]
		overlap_types_actual		=	[]
		overlap_offsets_predicted	=	[]
		overlap_types_predicted		=	[]
		
		
		for j in range(len(prediction[i])):
			pred_value	=	prediction[i][j]
			actu_value	=	y_values[i][j]
			if actu_value != 'NONE':
				bot_count	+=	1
				if actu_value not in class_dict:
					class_dict[actu_value]	=	{'TP':0,'FP':0,'FN':0}
				if pred_value not in class_dict:
					class_dict[pred_value]	=	{'TP':0,'FP':0,'FN':0}
				if pred_value == actu_value:
					top_count	+=	1
					class_dict[pred_value]['TP']	+=	1
				else:
					class_dict[pred_value]['FP']	+=	1
					class_dict[actu_value]['FN']	+=	1
			######################
			
			### RELAXED MATCH ###
			
			if actu_value != 'NONE':
				if actu_value != previous_actual:
					overlap_offsets_actual.append([i,i])
					overlap_types_actual.append(actu_value)
				else:
					overlap_offsets_actual[-1][-1]	+=	1
			previous_actual	=	actu_value
			
			if pred_value != 'NONE':
				if pred_value != previous_predicted:
					overlap_offsets_predicted.append([i,i])
					overlap_types_predicted.append(pred_value)
				else:
					overlap_offsets_predicted[-1][-1]	+=	1
			previous_predicted	=	pred_value
		
		###########
		
		### RELAXED MATCH ###
		
		# RECALL
		
		for j in range(len(overlap_offsets_actual)):
			found	=	False
			type	=	overlap_types_actual[j]
			if type not in type_counts:
				type_counts[type]	=	{'TP':0,'FP':0,'FN':0}
			if type not in iob_counts:
				iob_counts[type]	=	{'TP':0,'FP':0,'FN':0}
			for k in range(len(overlap_offsets_predicted)):
				a	=	overlap_offsets_actual[j][0]
				b	=	overlap_offsets_actual[j][1]
				c	=	overlap_offsets_predicted[k][0]
				d	=	overlap_offsets_predicted[k][1]
				if (a >= c and a <= d) or (b >= c and b <= d):
					found	=	True
					if overlap_types_actual[j] == overlap_types_predicted[k]:
						# overlap_top_count	+=	1
						type_counts[type]['TP']	+=	1
					else:
						type_counts[type]['FN']	+=	1
					break
			if found:
				# type	=	overlap_types_actual[j]
				iob_counts[type]['TP']	+=	1
			else:
				iob_counts[type]['FN']	+=	1
		
		# PRECISION
		
		for j in range(len(overlap_offsets_predicted)):
			found	=	False
			type	=	overlap_types_predicted[j]
			if type not in type_counts:
				type_counts[type]	=	{'TP':0,'FP':0,'FN':0}
			if type not in iob_counts:
				iob_counts[type]	=	{'TP':0,'FP':0,'FN':0}
			for k in range(len(overlap_offsets_actual)):
				a	=	overlap_offsets_predicted[j][0]
				b	=	overlap_offsets_predicted[j][1]
				c	=	overlap_offsets_actual[k][0]
				d	=	overlap_offsets_actual[k][1]
				if (a >= c and a <= d) or (b >= c and b <= d):
					found	=	True
					if overlap_types_predicted[j] == overlap_types_actual[k]:
						# overlap_top_count	+=	1
						type_counts[type]['TP']	+=	1
					else:
						type_counts[type]['FP']	+=	1
					break
			if found:
				# type	=	overlap_types_actual[j]
				iob_counts[type]['TP']	+=	1
			else:
				iob_counts[type]['FP']	+=	1
			
	all_p			=	[]
	all_r			=	[]
	
	sum_tp			=	0
	sum_fp			=	0
	sum_fn			=	0
	
	# RELAXED MATCH
	
	iob_all_p		=	[]
	iob_all_r		=	[]
	
	iob_sum_tp		=	0
	iob_sum_fp		=	0
	iob_sum_fn		=	0
	
	type_all_p		=	[]
	type_all_r		=	[]
	
	type_sum_tp		=	0
	type_sum_fp		=	0
	type_sum_fn		=	0
	
	print(top_count/bot_count)
	print('-'*97)
	print(f'{"==== PERFECT MATCH =====":>43} | ================== RELAXED MATCH ==================')
	print(f'{"| ========= IOB ========== | ========= TYPE =========":>97}')
	print(f'{"CLASS":<16} | {"PREC.":<8}{"REC.":<8}{"SUP.":<8} | {"PREC.":<8}{"REC.":<8}{"SUP.":<8} | {"PREC.":<8}{"REC.":<8}{"SUP.":<8}')
	for j in class_dict:
		if j == 'NONE':
			continue
		sum_tp	+=	class_dict[j]['TP']
		sum_fp	+=	class_dict[j]['FP']
		sum_fn	+=	class_dict[j]['FN']
		
		try:
			precision	=	class_dict[j]['TP'] / (class_dict[j]['TP']+class_dict[j]['FP'])
		except ZeroDivisionError as e:
			precision	=	'N/A'
		if precision != 'N/A':
			all_p.append(precision)
		
		try:
			recall		=	class_dict[j]['TP'] / (class_dict[j]['TP']+class_dict[j]['FN'])
		except ZeroDivisionError as e:
			recall		=	'N/A'
		if recall != 'N/A':
			all_r.append(recall)
		
		# RELAXED
		
		# IOB
		iob_sum_tp	+=	iob_counts[j]['TP']
		iob_sum_fp	+=	iob_counts[j]['FP']
		iob_sum_fn	+=	iob_counts[j]['FN']
		
		try:
			iob_precision	=	iob_counts[j]['TP'] / (iob_counts[j]['TP']+iob_counts[j]['FP'])
		except ZeroDivisionError as e:
			iob_precision	=	'N/A'
		if iob_precision != 'N/A':
			iob_all_p.append(iob_precision)
		
		try:
			iob_recall		=	iob_counts[j]['TP'] / (iob_counts[j]['TP']+iob_counts[j]['FN'])
		except ZeroDivisionError as e:
			iob_recall		=	'N/A'
		if iob_recall != 'N/A':
			iob_all_r.append(iob_recall)
		
		
		# TYPE
		type_sum_tp	+=	type_counts[j]['TP']
		type_sum_fp	+=	type_counts[j]['FP']
		type_sum_fn	+=	type_counts[j]['FN']
		
		try:
			type_precision	=	type_counts[j]['TP'] / (type_counts[j]['TP']+type_counts[j]['FP'])
		except ZeroDivisionError as e:
			type_precision	=	'N/A'
		if type_precision != 'N/A':
			type_all_p.append(type_precision)
		
		try:
			type_recall		=	type_counts[j]['TP'] / (type_counts[j]['TP']+type_counts[j]['FN'])
		except ZeroDivisionError as e:
			type_recall		=	'N/A'
		if type_recall != 'N/A':
			type_all_r.append(type_recall)
		
		# print(f'{j:<20}{precision:<20}{recall:<20}{class_dict[j]["TP"]+class_dict[j]["FN"]:>10}')
		# print(f'{j:<16}{precision:.4f<8}{recall:.4f<8}{class_dict[j]["TP"]+class_dict[j]["FN"]:>8}')
		print(f'{j:<16} | {precision:.5f} {recall:.5f} {class_dict[j]["TP"]+class_dict[j]["FN"]:>8} | {iob_precision:.5f} {iob_recall:.5f} {iob_counts[j]["TP"]+iob_counts[j]["FN"]:>8} | {type_precision:.5f} {type_recall:.5f} {type_counts[j]["TP"]+type_counts[j]["FN"]:>8}')
	
	print('-'*97)
	
	macro_p		=	sum(all_p)/len(all_p) if len(all_p) > 0 else 'N/A'
	macro_r		=	sum(all_r)/len(all_r) if len(all_r) > 0 else 'N/A'
	macro_f1	=	(2*macro_p*macro_r)/(macro_p+macro_r) if (macro_p != 'N/A' and macro_r != 'N/A') else 'N/A'
	
	micro_p		=	sum_tp	/	(sum_tp+sum_fp) if (sum_tp+sum_fp) > 0 else 'N/A'
	micro_r		=	sum_tp	/	(sum_tp+sum_fn) if (sum_tp+sum_fn) > 0 else 'N/A'
	micro_f1	=	(2*micro_p*micro_r)/(micro_p+micro_r) if (micro_p != 'N/A' and micro_r != 'N/A') else 'N/A'
	
	
	
	
	iob_macro_p		=	sum(iob_all_p)/len(iob_all_p) if len(iob_all_p) > 0 else 'N/A'
	iob_macro_r		=	sum(iob_all_r)/len(iob_all_r) if len(iob_all_r) > 0 else 'N/A'
	iob_macro_f1	=	(2*iob_macro_p*iob_macro_r)/(iob_macro_p+iob_macro_r) if (iob_macro_p != 'N/A' and iob_macro_r != 'N/A') else 'N/A'
	
	iob_micro_p		=	iob_sum_tp	/	(iob_sum_tp+iob_sum_fp) if (iob_sum_tp+iob_sum_fp) > 0 else 'N/A'
	iob_micro_r		=	iob_sum_tp	/	(iob_sum_tp+iob_sum_fn) if (iob_sum_tp+iob_sum_fn) > 0 else 'N/A'
	iob_micro_f1	=	(2*iob_micro_p*iob_micro_r)/(iob_micro_p+iob_micro_r) if (iob_micro_p != 'N/A' and iob_micro_r != 'N/A') else 'N/A'
	
	
	
	
	type_macro_p		=	sum(type_all_p)/len(type_all_p) if len(type_all_p) > 0 else 'N/A'
	type_macro_r		=	sum(type_all_r)/len(type_all_r) if len(type_all_r) > 0 else 'N/A'
	type_macro_f1	=	(2*type_macro_p*type_macro_r)/(type_macro_p+type_macro_r) if (type_macro_p != 'N/A' and type_macro_r != 'N/A') else 'N/A'
	
	type_micro_p		=	type_sum_tp	/	(type_sum_tp+type_sum_fp) if (type_sum_tp+type_sum_fp) > 0 else 'N/A'
	type_micro_r		=	type_sum_tp	/	(type_sum_tp+type_sum_fn) if (type_sum_tp+type_sum_fn) > 0 else 'N/A'
	type_micro_f1	=	(2*type_micro_p*type_micro_r)/(type_micro_p+type_micro_r) if (type_micro_p != 'N/A' and type_micro_r != 'N/A') else 'N/A'
	
	print(f'{"MACRO-PRECISION":<16} | {macro_p:<24} | {iob_macro_p:<24} | {type_macro_p:<24}')
	print(f'{"MACRO-RECALL":<16} | {macro_r:<24} | {iob_macro_r:<24} | {type_macro_r:<24}')
	print(f'{"MACRO-F1":<16} | {macro_f1:<24} | {iob_macro_f1:<24} | {type_macro_f1:<24}')
	
	print(f'{"MICRO-PRECISION":<16} | {micro_p:<24} | {iob_micro_p:<24} | {type_micro_p:<24}')
	print(f'{"MICRO-RECALL":<16} | {micro_r:<24} | {iob_micro_r:<24} | {type_micro_r:<24}')
	print(f'{"MICRO-F1":<16} | {micro_f1:<24} | {iob_micro_f1:<24} | {type_micro_f1:<24}')
	
	print(f'{"COHEN KAPPA":<16} | {cohen_score:<24} | {"":<24} | {"":<24}')
	
	print('-'*97)
	
	config['stats']	=	{
		'macro_p'	:	macro_p,
		'macro_r'	:	macro_r,
		'macro_f1'	:	macro_f1,
		'micro_p'	:	micro_p,
		'micro_r'	:	micro_r,
		'micro_f1'	:	micro_f1,
		'iob_macro_p'	:	iob_macro_p,
		'iob_macro_r'	:	iob_macro_r,
		'iob_macro_f1'	:	iob_macro_f1,
		'iob_micro_p'	:	iob_micro_p,
		'iob_micro_r'	:	iob_micro_r,
		'iob_micro_f1'	:	iob_micro_f1,
		'type_macro_p'	:	type_macro_p,
		'type_macro_r'	:	type_macro_r,
		'type_macro_f1'	:	type_macro_f1,
		'type_micro_p'	:	type_micro_p,
		'type_micro_r'	:	type_micro_r,
		'type_micro_f1'	:	type_micro_f1
	}
	"""
	
	config["training"]	=	False
	
	json_config_path	=	f'{config["model_path"][:-3]}_config.json'
	try:
		new_config_file	=	open(json_config_path,'wt',encoding='UTF-8')
		json.dump(config, new_config_file, indent=8)
		new_config_file.close()
	except IOError or OSError as e:
		print(e)
		print(f'WARNING: COULD NOT SAVE CONFIGURATION FILE TO PATH {json_config_path}, SEE MESSAGE JUST ABOVE')
	else:
		print(f'Saved configuration file in {json_config_path}')
	
# FROM OFFICIAL DOCUMENTATION
def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print("%0.6f %-8s %s" % (weight, label, attr))

def main() -> int:
	global config
	try:
		config_file	=	open(config_path,'rt',encoding='UTF-8')
		config		=	json.load(config_file)
		config_file.close()
	except IOError or OSError as e:
		print(e)
		print(f'Could not start script without configuration file, exitting')
		exit()
	print(f'Loaded configuration from file {config_path}')
	
	
	global CRF_model
	prepare_model()

	if config["training"]:
		train()
	
	estimate_performance_on_dev()
	
	# print("Top positive:")
	# print_state_features(Counter(CRF_model.state_features_).most_common(20))
			
	return 0

########################################

if __name__ == '__main__':
	t		=	time.time()
	main_result	=	main()
	print(f'Program ended with result {main_result} in {time.time()-t}s')
