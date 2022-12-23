"""
AUTHOR		:	UNIVERSITY OF ORLEANS (FRANCE)
TASK		:	LEGALEVAL2023
SUB-TASK	:	L-NER
"""

import time
import re
import spacy
from os import sys
from os import listdir
from os.path import exists
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import json
import sklearn_crfsuite
from sklearn.metrics import accuracy_score, cohen_kappa_score
from seqeval.metrics.sequence_labeling import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2, IOBES
import json

from L_NER_CRF_train import post_traitement, IOB2_transformer, IOBES_transformer

########################################

config_path		=	'L_NER_CRF_default_config.json'
# config_path		=	'L_NER_CRF_model_100_config.json'

# https://readthedocs.org/projects/sklearn-crfsuite/downloads/pdf/latest/

########################################

config				=	None
spacy_docs			=	[]

########################################

anti_whitespace_regex		=	re.compile('\\S')

date_regex			=	re.compile('((0?\\d|[12]\\d|3[01])(?P<SEP1>\\D)(0?\\d|1[012])(?P=SEP1)(18|19|20)\\d{2})|((0?\\d|1[012])(?P<SEP2>\\D)(0?\\d|[12]\\d|3[01])(?P=SEP2)(18|19|20)\\d{2})|((18|19|20)\\d{2}(?P<SEP3>\\D)(0?\\d|[12]\\d|3[01])(?P=SEP3)(0?\\d|1[012]))|((18|19|20)\\d{2})(?P<SEP4>\\D)(0?\\d|1[012])(?P=SEP4)(0?\\d|[12]\\d|3[01])')
org_regex			=	re.compile('[A-Z]{3,}')

CRF_model			=	None
annot_df			=	None

def generate_features_from_SpaCy_doc(
		input_doc	:	spacy.tokens.Doc
	):
	'''
	This function is to be called to generate features from a raw str, with SpaCy.
	
	def generate_features_from_SpaCy_doc(
			input_doc	:	spacy.tokens.Doc
		):
		[...]
	'''
	########
	
	global config
	
	if 'spacy' not in dir():
		import spacy
		nlp	=	spacy.load(config["spacy_model"])
	
	X_features	=	[]
	X_offsets	=	[]
	
	# for j in doc:
	for j in input_doc:
		if anti_whitespace_regex.search(j.text) == None:
			continue
		features	=	{}
		for k in config["relevant_features"]:
			attr			=	getattr(j,k)
			attr			=	attr if '__call__' not in dir(attr) else attr()
			features[k.lstrip('_')]	=	attr if type(attr) in [bool,int,str] else str(attr)
		X_features.append(features)
		
		tok_start		=	j.idx
		tok_end			=	j.idx + len(j)
		
		X_offsets.append((tok_start,tok_end))
	return X_features, X_offsets

def features_for_CRF_from_SpaCy_doc(
		input_doc	:	spacy.tokens.Doc
	):
	'''
	This function returns features and offsets from a raw str, using SpaCy.
	Both are in a separate list.	
	
	def features_for_CRF_from_SpaCy_doc(
			input_doc	:	spacy.tokens.Doc
		):
		[...]
	'''
	########
	
	# X_features, X_offsets	=	generate_features_from_str(input_str)
	X_features, X_offsets	=	generate_features_from_SpaCy_doc(input_doc)
	if __debug__:
		assert len(X_features) == len(X_offsets)
	
	if "drop_rows_if_true" in config:
		# len_df	=	len(features_df)
		
		len_X_features	=	len(X_features)
		
		new_X_features	=	[]
		new_X_offsets	=	[]
		
		for j in range(len(X_features)):
			appending	=	True
			for i in config["drop_rows_if_true"]:
				if i in X_features[j] and X_features[j][i] == True:
					appending	=	False
					break
			if appending:
				new_X_features.append(X_features[j])
				new_X_offsets.append(X_offsets[j])

		# print(f'Dropped {len_X_features-len(new_X_features)}',end="\r")
		
		X_features	=	new_X_features
		X_offsets	=	new_X_offsets
	
	if __debug__:
		assert len(X_features) == len(X_offsets)
	
	for i in config["irrelevant_features"]:
		for j in range(len(X_features)):
			if i in X_features[j]:
				X_features[j].pop(i)
	
	look_behind	=	config["look_behind"]
	look_ahead	=	config["look_ahead"]
	len_X_features	=	len(X_features)
	for j in range(len_X_features):
		for k in range(j-look_behind,j+look_ahead+1):
			if k < 0 or k >= len_X_features or k == j:
				continue
			prefix	=	f"{'+' if k > j else ''}{k-j}:"
			for m in X_features[k]:
				if m[0] in ['-','+']:
					continue
				X_features[j][f"{prefix}{m}"]	=	X_features[k][m]
	
	if __debug__:
		assert len(X_features) == len(X_offsets)
	
	return X_features, X_offsets

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

# FROM OFFICIAL DOCUMENTATION
def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print("%0.6f %-8s %s" % (weight, label, attr))

def main() -> int:
	corpus_objects	=	{}
	
	i	=	0
	i_limit	=	len(sys.argv)
	while i < i_limit:
		# if sys.argv[i] == "-f":
		if sys.argv[i] == "--f" or sys.argv[i] == "--file":
			i	+=	1
			if sys.argv[i].endswith('.json'):
				try:
					file				=	open(sys.argv[i],"rt",encoding="UTF-8")
					corpus_content			=	json.load(file)
					corpus_objects[sys.argv[i]]	=	corpus_content
					file.close()
				except IOError or OSError as e:
					print(e)
					print(f"Could not open file {sys.argv[i]}. See error message above. Exiting.")
					exit()
			else:
				print(f"Given file ({sys.argv[i]}) is not a JSON file. Exiting.")
				exit()
		# elif sys.argv[i] == "-d":
		elif sys.argv[i] == "--d" or sys.argv[i] == "--directory":
			i	+=	1
			ld	=	listdir(sys.argv[i])
			for j in ld:
				if (not j.endswith('.json')) or (j.endswith('_OUTPUT.json')):
					continue
				local_path	=	f"{sys.argv[i]}{'/' if not sys.argv[i].endswith('/') else ''}{j}"
				try:
					file				=	open(local_path)
					corpus_content			=	json.load(file)
					corpus_objects[local_path]	=	corpus_content
					file.close()
				except IOError or OSError as e:
					print(e)
					print(f"Could not open file {local_path}. See error message above. Ignoring.")
		i	+=	1
	
	print(f"Found {len(corpus_objects)} elements :")
	for i in corpus_objects:
		print(i)
	
	global config
	try:
		config_file	=	open(config_path,'rt',encoding='UTF-8')
		config		=	json.load(config_file)
		config_file.close()
	except IOError or OSError as e:
		print(e)
		print(f'Could not start script without configuration file. Exiting.')
		exit()
	print(f'Loaded configuration from file {config_path}')
	
	global CRF_model
	# prepare_model()
	
	try:
		CRF_model	=	joblib.load(config["model_path"])
		print(f'Loaded {config["model_path"]}')
	except OSError or IOError as e:
		print(e)
		print(f'Could not load model from {config["model_path"]}, see message above. Exiting.')
		return 1
	
	global spacy_docs
	global doc_index
	
	nlp	=	spacy.load(config["spacy_model"])
	print(f'Loaded SpaCy with model {config["spacy_model"]}')
	
	annot_count	=	0
	file_count	=	1
	n_files		=	len(corpus_objects)
	for i in corpus_objects:
		###########################
		doc_index	=	0
		text_list	=	[]
		id_list		=	[]
		# text_dict	=	{}
		for j in corpus_objects[i]:
			text_list.append(j["data"]["text"])
			id_list.append(j["id"])
			# text_dict[j["id"]]	=	j["data"]["text"]
		# annotation_docs	=	[]
		annotation_docs	=	{}
		len_text_list	=	len(text_list)
		print('SpaCy processing:')
		for doc in nlp.pipe(text_list):
			print(f'File {file_count}/{n_files} ({i}); text {doc_index+1}/{len_text_list} ({id_list[doc_index]})',end='\r')
			# annotation_docs.append(doc)
			annotation_docs[id_list[doc_index]]	=	doc
			doc_index	+=	1
		# doc_index	=	0
		###########################
		
		X_features_full	=	[]
		X_offsets_full	=	[]
		
		len_i	=	len(corpus_objects[i])
		j_count	=	1
		
		print('\nFeature generation:')
		for j in corpus_objects[i]:
			# print(f'{j_count:>6} / {len_i} ; id={j["id"]}',end="\r")
			print(f'File {file_count}/{n_files} ({i}); text {j_count}/{len_i} ({j["id"]})',end='\r')
			
			j["annotations"]	=	[]
			
			text	=	j["data"]["text"]
			# X_features, X_offsets	=	features_for_CRF_from_str(text)
			X_features, X_offsets	=	features_for_CRF_from_SpaCy_doc(annotation_docs[j["id"]])
			if __debug__:
				assert len(X_features) == len(X_offsets)
			
			X_features_full.append(X_features)
			X_offsets_full.append(X_offsets)
			
			# prediction	=	CRF_model.predict([X_features])[0]
			j_count	+=	1	#!
		j_count	=	1
		
		prediction_full	=	CRF_model.predict(X_features_full)
		prediction_full	=	post_traitement(
			X_features		=	X_features_full,
			prediction		=	prediction_full,
			enable_cities_query	=	"enable_cities_query" in config and config["enable_cities_query"],
			enable_DATE_regex	=	"enable_DATE_regex" in config and config["enable_DATE_regex"],
			enable_ORG_regex	=	"enable_ORG_regex" in config and config["enable_ORG_regex"]
		)
		
		print('\nPrediction:')
		for j in corpus_objects[i]:
			# print(f'{j_count:>6} / {len_i} ; id={j["id"]}',end="\r")
			print(f'File {file_count}/{n_files} ({i}); text {j_count}/{len_i} ({j["id"]})',end='\r')
			
			result_obj	=	{
				"result"	:	[]
			}
			
			text		=	j["data"]["text"]
			prediction	=	prediction_full[j_count-1]
			X_offsets	=	X_offsets_full[j_count-1]
			
			start_index	=	None
			end_index	=	None
			NE_type		=	None
			for k in range(len(prediction)):
				if prediction[k].startswith('B-') or prediction[k].startswith('S-') or prediction[k] == 'O':
					if start_index != None:
						value_obj	=	{
							"start"		:	start_index,
							"end"		:	end_index,
							"text"		:	text[start_index:end_index],
							"labels"	:	[
								NE_type
							]
						}
						result_obj["result"].append({
							"value"		:	value_obj,
							"id"		:	f"ANNOT_{annot_count}",
							"from_name"	:	"label",
							"to_name"	:	"text",
							"type"		:	"labels"
						})
						start_index	=	None
						end_index	=	None
						NE_type		=	None
						annot_count	+=	1
				if prediction[k].startswith('B-') or prediction[k].startswith('S-'):
					start_index	=	X_offsets[k][0]
					end_index	=	X_offsets[k][1]
					NE_type		=	prediction[k][2:]
				
				elif prediction[k].startswith('I-') or prediction[k].startswith('E-'):
					end_index	=	X_offsets[k][1]
			
			if start_index != None:
				value_obj	=	{
					"start"		:	start_index,
					"end"		:	end_index,
					"text"		:	text[start_index:end_index],
					"labels"	:	[
						NE_type
					]
				}
				result_obj["result"].append({
					"value"		:	value_obj,
					"id"		:	f"ANNOT_{annot_count}",
					"from_name"	:	"label",
					"to_name"	:	"text",
					"type"		:	"labels"
				})
				start_index	=	None
				end_index	=	None
				NE_type		=	None
				annot_count	+=	1

			j["annotations"].append(result_obj)
			j_count	+=	1

		output_path	=	f"{i[:-5]}_OUTPUT.json"
		try:
			output_file	=	open(output_path,"wt",encoding="UTF-8")
			json.dump(corpus_objects[i],output_file,indent=8)
			print(f"\nWrote {output_path}")
		except IOError or OSError as e:
			print(e)
			print(f"\nCould not output to file {output_path}. See message above.")
			
		output_file.close()
		
		file_count	+=	1
	
	# print("Top positive:")
	# print_state_features(Counter(CRF_model.state_features_).most_common(20))
			
	return 0

########################################

if __name__ == '__main__':
	if __debug__:
		print("Debugging is ON")
	else:
		print("Debugging is OFF")
	t		=	time.time()
	main_result	=	main()
	print(f'Program ended with result {main_result} in {time.time()-t}s')