"""
AUTHOR		:	UNIVERSITY OF ORLEANS (FRANCE)
TASK		:	LEGALEVAL2023
SUB-TASK	:	L-NER
***************************************************************
FUNCTIONS DEFINED IN THIS SCRIPT
- generate_features_from_SpaCy_doc
- features_for_CRF_from_SpaCy_doc
- regex_annotation
- post_processing_from_raw_offsets
- levenshtein_distance
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
from seqeval.metrics.sequence_labeling import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2, IOBES
import json

from L_NER_CRF_train import IOB2_transformer, IOBES_transformer

########################################

# config_path		=	'L_NER_CRF_default_config.json'
# config_path		=	'L_NER_CRF_model_100_config.json'
# config_path		 =   'L_NER_CRF_model_106_MERGED_config_NO_POSTPROCESSING.json'
# config_path		 =   'L_NER_CRF_model_106_MERGED_config_POSTPROCESSING_DATE.json'
# config_path		 =   'L_NER_CRF_model_106_MERGED_config_POSTPROCESSING_DATE_FIX.json'
config_path		 =   'L_NER_CRF_model_106_MERGED_config_DATE_ORG_GPE_CASE_NUMBER_POSTPROCESSING.json'

# https://readthedocs.org/projects/sklearn-crfsuite/downloads/pdf/latest/

########################################

config				=	None
spacy_docs			=	[]

########################################

anti_whitespace_regex		=	re.compile('\\S')

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
	
	X_features, X_offsets	=	generate_features_from_SpaCy_doc(input_doc)
	if __debug__:
		assert len(X_features) == len(X_offsets)
	
	if "drop_rows_if_true" in config:
		len_X_features	=	len(X_features)
		
		new_X_features	=	[]
		new_X_offsets	=	[]
		
		for j in range(len_X_features):
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
		
	len_X_features	=	len(X_features)
	
	for i in config["irrelevant_features"]:
		for j in range(len_X_features):
			if i in X_features[j]:
				X_features[j].pop(i)
	
	look_behind	=	config["look_behind"]
	look_ahead	=	config["look_ahead"]
	for j in range(len_X_features):	
		for k in range(-look_behind,look_ahead+1):
			if k == 0 or j+k < 0 or j+k >= len_X_features:
				continue
			prefix	=	f"{'+' if k > 0 else ''}{k}:"
			for m in X_features[j+k]:
				if m[0] in ['-','+']:
					continue
				X_features[j][f"{prefix}{m}"]	=	X_features[j+k][m]
	
	if __debug__:
		assert len(X_features) == len(X_offsets)
	
	return X_features, X_offsets

def regex_annotation(
		raw_texts   :   list,
		regex_pattern   :   str
	):
	'''
	Returns the annotation spans for a given regex over a number of texts.

	def regex_annotation(
			raw_texts   :   list,
			regex_pattern   :   str
		):
	'''
	results =   []
	local_regex	=	re.compile(regex_pattern)
	for i in range(len(raw_texts)):
		results.append([])
		for j in local_regex.finditer(raw_texts[i]):
			results[i].append((j.start(),j.end()))
	return results
	

def post_processing_from_raw_offsets(
		raw_texts		:	list,
		enable_cities_query	:	bool	=	False,
		enable_DATE_regex	:	bool	=	False,
		enable_ORG_regex	:	bool	=	False,
		enable_CASE_NUMBER_regex	:	bool	=	False
	) -> list:
	'''
	This function takes the features and predictions, and returns the predictions once post-processing has been done.
	
	
	def post_processing_from_raw_offsets(
			raw_texts			:	list,
			enable_cities_query	:	bool	=	False,
			enable_DATE_regex	:	bool	=	False,
			enable_ORG_regex	:	bool	=	False,
			enable_CASE_NUMBER_regex	:	bool	=	False
		) -> list:
	'''
	
	output_annotations	=	[{} for i in raw_texts]
	
	if enable_cities_query:
		try:
			cities_df		=	pd.read_csv('cities.csv' if "cities_path" not in config else config["cities_path"],sep='\t',encoding='UTF-8')
		except OSError or IOError as e:
			print(e)
			print("Could not open path to cities file. See message above.")
		else:
			cities_df['city']	=	cities_df['city'].str.lower()
			cities_df['state']	=	cities_df['state'].str.lower()
			cities_and_states	=	cities_df['city'].tolist() + cities_df['state'].tolist()
			
			for i in range(len(raw_texts)):
				# i	=	i.lower()
				i_lower	=	raw_texts[i].lower()
				if 'GPE' not in output_annotations[i]:
					output_annotations[i]['GPE']	=	[]
				for j in cities_and_states:
					local_find	=	i_lower.find(j)
					if local_find >= 0:
						output_annotations[i]['GPE'].append((local_find,local_find+len(j)))
	
	
	
	if enable_DATE_regex:
		if "DATE_regex_pattern" in config:
			regex_result	=   regex_annotation(raw_texts,config["DATE_regex_pattern"])
			for i in range(len(raw_texts)):
				output_annotations[i]['DATE']   =   regex_result[i]
		else:
			print("Could not find \"DATE_regex_pattern\" in config file, or it is not of type str.")
		
	
	if enable_ORG_regex:
		if "ORG_regex_pattern" in config:
			regex_result	=   regex_annotation(raw_texts,config["ORG_regex_pattern"])
			for i in range(len(raw_texts)):
				output_annotations[i]['ORG']   =   regex_result[i]
		else:
			print("Could not find \"ORG_regex_pattern\" in config file, or it is not of type str.")
	
	if enable_CASE_NUMBER_regex:
		if "CASE_NUMBER_regex_pattern" in config:
			regex_result	=   regex_annotation(raw_texts,config["CASE_NUMBER_regex_pattern"])
			for i in range(len(raw_texts)):
				output_annotations[i]['CASE_NUMBER']   =   regex_result[i]
		else:
			print("Could not find \"CASE_NUMBER_regex_pattern\" in config file, or it is not of type str.")

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
	
	return output_annotations


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
	'''
	C-like equivalent of the main function.
	Output :
		0 -> no error
		1 -> error

	def main() -> int:
		[...]
	'''
	###
	
	########
	# Read content from files given in command line
	########
	
	corpus_objects	=	{}
	
	i	=	0
	i_limit	=	len(sys.argv)
	while i < i_limit:
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
	
	########
	# Load configuration file that's necessary for the model
	########
	
	global config
	try:
		config_file	=	open(config_path,'rt',encoding='UTF-8')
		config		=	json.load(config_file)
		config_file.close()
		print(f'Loaded configuration from file {config_path}')
	except IOError or OSError as e:
		print(e)
		print(f'Could not start script without configuration file. Exiting.')
		return 1
	
	
	########
	# For each file given as input, process it and generate an output
	########
	
	global spacy_docs
	global doc_index

	annot_count	=	0
	file_count	=	1
	n_files		=	len(corpus_objects)
	for i in corpus_objects:
		###########################
		doc_index	=	0
		text_list	=	[]
		id_list		=	[]
		for j in corpus_objects[i]:
			text_list.append(j["data"]["text"])
			id_list.append(j["id"])
		annotation_docs	=	{}
		len_text_list	=	len(text_list)
		
		###########################
		
		len_i		=	len(corpus_objects[i])
		j_count		=	1
		
		########
		# Transform IOB2 annotations into raw offset annotations and place them into the JSON structure
		########
		
		occupied_offsets	=	{}
		
		for j in corpus_objects[i]:
			occupied_offsets[j["id"]]	=	[]
			print(f'File {file_count}/{n_files} ({i}); text {j_count}/{len_i} ({j["id"]})',end='\r')
			
			for k in j["annotations"][0]["result"]:
				occupied_offsets[j["id"]].append((k["value"]["start"],k["value"]["end"]))

			j_count	+=	1
		
		########
		# Launch post processing from raw offsets
		########
		
		post_processing_annotations	=	post_processing_from_raw_offsets(
			raw_texts		=	text_list,
			enable_cities_query	=	"enable_cities_query" in config and config["enable_cities_query"],
			enable_DATE_regex	=	"enable_DATE_regex" in config and config["enable_DATE_regex"],
			enable_ORG_regex	=	"enable_ORG_regex" in config and config["enable_ORG_regex"],
			enable_CASE_NUMBER_regex	=	"enable_CASE_NUMBER_regex" in config and config["enable_CASE_NUMBER_regex"]
		)
		post_processing_added_annotations_count	=	0
		for j in range(len(post_processing_annotations)):
			for k in post_processing_annotations[j]:
				for m in range(len(post_processing_annotations[j][k])):
					if post_processing_annotations[j][k][m] not in occupied_offsets[id_list[j]]:
						start_index	=	post_processing_annotations[j][k][m][0]
						end_index	=	post_processing_annotations[j][k][m][1]
						value_obj	=	{
							"start"		:	start_index,
							"end"		:	end_index,
							"text"		:	text_list[j][start_index:end_index],
							"labels"	:	[
								k
							]
						}
						#result_obj["result"].append({
						#corpus_objects[i]["annotations"][0]["result"].append({
						corpus_objects[i][j]["annotations"][0]["result"].append({
							"value"		:	value_obj,
							"id"		:	f"ANNOT_{annot_count}",
							"from_name"	:	"label",
							"to_name"	:	"text",
							"type"		:	"labels"
						})
						annot_count	+=	1
						occupied_offsets[id_list[j]].append((start_index,end_index))
						post_processing_added_annotations_count	+=	1
		print(f'\nPost-processing added {post_processing_added_annotations_count} annotations')
		
		########
		# Write the new file
		########
		
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
