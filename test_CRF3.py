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
from sklearn.metrics import accuracy_score

########################################

config_path		=	'default_CRF_config.json'

# https://readthedocs.org/projects/sklearn-crfsuite/downloads/pdf/latest/

########################################

try:
	config_file	=	open(config_path,'rt',encoding='UTF-8')
	config		=	json.load(config_file)
	config_file.close()
except IOError or OSError as e:
	print(e)
	print(f'Could not start script without configuration file, exitting')
	exit()
print(f'Loaded configuration from file {config_path}')

########################################

anti_whitespace_regex		=	re.compile('\\S')

CRF_model			=	None
annot_df			=	None

def prepare_model():
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
			algorithm			=	config["solver"],
			c1				=	None if config["solver"].lower()!='lbfgs' else config["c1"],
			c2				=	None if config["solver"].lower() not in ['lbfgs','l2sgd'] else config["c2"],
			max_iterations			=	config["max_epoch"],
			gamma				=	None if config["solver"].lower()!='arow' else config["gamma"],
			all_possible_transitions	=	config["all_possible_transitions"],
			all_possible_states		=	config["all_possible_states"],
			variance			=	None if config["solver"].lower()!='arow' else config["variance"],
			epsilon				=	None if config["solver"].lower()=='l2sgd' else config["epsilon"],
			verbose				=	True,
			min_freq			=	0.0 if "min_freq" not in config else config["min_freq"]
		)
	return 0

def generate_features(text_base_path: str, features_memory_path: str):
	preamble_df	=	pd.read_csv(annot_preamble_path, encoding=config["encoding"], sep=config["sep"])
	preamble_df.dropna(how='all',inplace=True)
	judgement_df	=	pd.read_csv(annot_judgement_path, encoding=config["encoding"], sep=config["sep"])
	judgement_df.dropna(how='all',inplace=True)
	
	annot_df	=	pd.concat((preamble_df,judgement_df))
	
	del(preamble_df)
	del(judgement_df)
	
	
	X_features	=	[]
	y_values	=	[]
	
	
	# MOVED FROM EARLIER
	text_base_df	=	pd.read_csv(text_base_path, encoding=config["encoding"], sep=config["sep"]
	# , nrows=3000
	)
	text_base_df.dropna(how='all',inplace=True)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\n','\n',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\t','\t',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\r','\r',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\"','"',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace("\\'","'",regex=False)
	
	
	if 'spacy' not in dir():
		import spacy
	nlp		=	spacy.load(config["spacy_model"])
	
	
	i_count	=	1
	i_limit	=	len(text_base_df)
	for i in text_base_df['id']:
		print(f'Generating features for text {i_count:>6} / {i_limit} : {i}',end='\r')
		doc		=	nlp(text_base_df.loc[(text_base_df['id'] == i)]['text'].to_list()[0])
		for j in doc:
			if anti_whitespace_regex.search(j.text) == None:
				continue
			features	=	{
				'text_id'	:	i,
				'id_in_text'	:	j.i#,
				# 'type'		:	dataset_type
			}
			for k in config["relevant_features"]:
				attr			=	getattr(j,k)
				attr			=	attr if '__call__' not in dir(attr) else attr()
				# features[k]		=	attr
				features[k.lstrip('_')]	=	attr
			X_features.append(features)
			
			tok_start		=	j.idx
			tok_end			=	j.idx + len(j)
			features['offset_start']	=	tok_start
			features['offset_end']		=	tok_end
			relevant_annotations	=	annot_df.loc[(annot_df['text_id'] == i) & (annot_df['annotation_start'] <= tok_start) & (annot_df['annotation_end'] >= tok_end)]
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

def features_and_values_for_CRF(text_base_path: str, features_memory_path: str):
	X_features	=	[]
	y_values	=	[]
	
	if not exists(features_memory_path):
		print('Features were not found, generating them...')
		features_df	=	generate_features(text_base_path, features_memory_path)
		features_df	=	features_df[[i.lstrip('_') for i in config["relevant_features"] if i not in config["irrelevant_features"]] + ['y_value','text_id']]
	else:
		features_df	=	pd.read_csv(features_memory_path,encoding=config["encoding"],sep=config["sep"], index_col='id', na_values='N/A',dtype=config["dtype"])[[i.lstrip('_') for i in config["relevant_features"] if i not in config["irrelevant_features"]] + ['y_value','text_id']].copy() #FIXED #IMPROVED
		print(f'Loaded features from {features_memory_path}')
	
	features_df.dropna(how='all',inplace=True)
	
	for i in config["labels_to_erase"]:
		features_df['y_value']	=	features_df['y_value'].str.replace(i,'NONE',regex=False)
	
	print(features_df.columns)
	
	
	
	###################################
	
	all_text_ids		=	features_df['text_id'].unique()#[:50]
	
	columns_to_consider	=	[i for i in features_df.columns if i not in ['y_value','text_id','offset_start','offset_end']]
	
	
	print(f'Generating new columns...')
	
	features_df_gb	=	features_df.groupby('text_id')
	
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
		# if i_count % 
		print(f'y_values : {i_count:<6} / {i_limit}',end='\r')
		y_values.append(group['y_value'].to_list())
		i_count	+=	1
	features_df.pop('y_value')
	features_df.pop('text_id')
	i_count	=	1
	for name, group in features_df_gb:
		print(f'X_features : {i_count:<6} / {i_limit}',end='\r')
		X_features.append(group.to_dict('records'))
		i_count	+=	1
	
	for i in X_features:
		for j in i:
			keys	=	list(j.keys())
			for k in keys:
				if j[k] == None or type(j[k]) not in [bool,int,str]:
					j.pop(k)
	del(features_df)
	return X_features, y_values
	
def train():
	global CRF_model
	if CRF_model == None:
		prepare_model()
	X_features, y_values	=	features_and_values_for_CRF(config["text_base_train_path"], config["features_memory_train_path"])
	
	# print(X_features[:2],y_values[:2])
	# print(y_values[:15])
	
	CRF_model.fit(X_features,y_values)
	if config["save_model"]:
		try:
			joblib.dump(CRF_model, config["model_path"])
		except IOError or OSError as e:
			print(e)
			print(f'WARNING: COULD NOT SAVE MODEL TO PATH {config["model_path"]}, SEE MESSAGE JUST ABOVE')
		else:
			print(f'Saved {config["model_path"]}')


def estimate_performance_on_dev():
	global CRF_model
	if CRF_model == None:
		print('No CRF_model available, cannot estimate performance on DEV')
		exit()
	X_features, y_values	=	features_and_values_for_CRF(config["text_base_dev_path"], config["features_memory_dev_path"])
	
	
	prediction	=	CRF_model.predict(X_features)
	
	# X_text_id	=	[]
	X_text_col		=	[]
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
	
	
	
	results_df	=	pd.DataFrame()
	results_df['y_value']	=	pd.Series(y_value_col,dtype='category')
	results_df['y_predict']	=	pd.Series(y_predict_col,dtype='category')
	results_df['X_text']	=	pd.Series(X_text_col,dtype='str')
	results_df.index.name	=	'id'
	results_df.to_csv('latest_results.csv',sep=config["sep"],encoding=config["encoding"])
	print(f'Wrote latest_results.csv')
		
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