import json
import pandas as pd
import numpy as np
# from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import classification_report
from seqeval.scheme import IOB2, IOBES
import matplotlib.pyplot as plt
import matplotlib.colors as colors

config_path	=	'L_NER_analysis_config.json'

'''
SUGGESTED RUNS:
- confusion_matrix_based_on_full_annotations -> true
- confusion_matrix_based_on_full_annotations -> false, remove_IOB -> true
- confusion_matrix_based_on_full_annotations -> false, remove_IOB -> false
'''

##################################################


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
	try:
		config_file	=	open(
			config_path,
			'rt',
			encoding='UTF-8'
		)
		config		=	json.load(config_file)
		config_file.close()
	except IOError or OSError as e:
		print(e)
		print(f'Could not open file {config_path}, exiting.')
		return 1

	##################################################
	
	########
	# Reading the results
	########
	try:
		df	=	pd.read_csv(
			config["result_path"],
			sep		=	config["result_separator"],
			encoding	=	config["result_encoding"],
			index_col	=	config["result_index_col"],
			dtype		=	config["result_dtype"]
		)
	except IOError or OSError as e:
		print(e)
		print(f'Could not open file {config["result_path"]}, exiting.')
		return 1
		
	########
	# Make a pair of 2D matrices, the first dimension being the texts, the second dimension being the tokens of these texts; y_value is the matrix for the right values, y_predict is the matrix for the predicted values
	########
	df_gb		=	df.groupby('text_id')
	y_value		=	[]
	y_predict	=	[]
	i_count		=	1
	i_max		=	len(df_gb)
	for name, group in df_gb:
		# print(f'Handling {name}',end='\r')
		print(f'Handling text {i_count}/{i_max}; {name}',end='\r')
		y_value.append(group['y_value'].to_list())
		y_predict.append(group['y_predict'].to_list())
		i_count	+=	1
	print(f'\n{"-"*64}')

	########
	# Make a classification report based on the pair of matrices we generated, and print it
	########
	cr_dict		=	classification_report(
					y_value,
					y_predict,
					output_dict	=	True,
					scheme		=	IOBES if config["IOBES"] else IOB2,
					mode		=	config["mode"]	# 'strict' OR 'relaxed'
				)

	df_results	=	pd.DataFrame.from_dict(cr_dict, orient='index')
	print(df_results)
	print(f'{"-"*64}')

	########
	# Make a plot for precision, recall, F1-score
	########
	axes		=	df_results[['precision','recall','f1-score']].plot.bar(
					rot		=	25,
					subplots	=	True,
					ylim		=	(0.0,1.0),
				)
	axes[1].legend(loc=2)

	plt.show()

	########
	# Make a confusion matrix based on the different classes, and then display it; it is strongly recommended to have "remove_IOB" set to true in the configuration file if confusion_matrix_based_on_full_annotations is set to false, mostly for readability of the plot
	# About confusion_matrix_based_on_full_annotations, if set to true, the classes considered most overlapping will be on the basis of entire annotations, and if set to false it will be on the basis of the number of tokens
	# remove_IOB has no effect if confusion_matrix_based_on_full_annotations is set to true, as IOB will be removed whatever happens in this option
	########
	df_ct	=	None
	if config["confusion_matrix_based_on_full_annotations"]:
		'''
		This perspective can be understood as : any continuous area of the same categories is counted as one overlap.
		For example:
		
		XXX XXXXX XXX XX XXXXXXX XXXX XX XXXXXX
		          ^^^^^^^^^^^^^^^^^^^
		     O             A             O
		     
		^^^^^^^^^^^^^            ^^^^^^^^^^^^^^
		     B             O             C
		
		In such a configuration, would be counted, in order:
		Overlap O/B
		Overlap A/B
		Overlap A/O
		Overlap A/C
		Overlap O/C
		'''
		local_dict	=	{}
		n_text_id	=	len(df['text_id'].unique())
		n		=	1
		# This is the same code as the one in the other option, for removing IOB, but here it is not dependent of config["remove_IOB"] because it's better to do it whatever happens, as we're handling full annotations
		df['y_value']	=	df['y_value'].apply(lambda x: x[2:] if x != 'O' else 'NONE')
		df['y_predict']	=	df['y_predict'].apply(lambda x: x[2:] if x != 'O' else 'NONE')
		for name, group in df.groupby('text_id'):
			print(f'Matrix {n}/{n_text_id}; {name}',end='\r')
			previous_y_value	=	'NONE'
			previous_y_predict	=	'NONE'
			for id, row in group.iterrows():
				new_y_value	=	row['y_value']
				new_y_predict	=	row['y_predict']
				
				'''
				if new_y_value == 'O':
					new_y_value	=	'NONE'
				else:
					new_y_value	=	new_y_value[2:]
					
				if new_y_predict == 'O':
					new_y_predict	=	'NONE'
				else:
					new_y_predict	=	new_y_predict[2:]
				'''
				
				if (new_y_value != previous_y_value) or (new_y_predict != previous_y_predict):
					if new_y_predict not in local_dict:
						local_dict[new_y_predict]	=	{}
					if new_y_value not in local_dict[new_y_predict]:
						local_dict[new_y_predict][new_y_value]	=	0
					local_dict[new_y_predict][new_y_value]	+=	1
				
				previous_y_value	=	new_y_value
				previous_y_predict	=	new_y_predict
			n	+=	1
		# df_ct	=	pd.DataFrame.from_dict(local_dict)
		df_ct	=	pd.DataFrame.from_dict(local_dict,orient='index')
	else:
		'''
		This perspective can be understood as : each token is individually considered as an overlap.
		For example:
		
		(1) (2)   (3) (4)(5)     (6)  (7)(8)
		XXX XXXXX XXX XX XXXXXXX XXXX XX XXXXXX
		          ^^^^^^^^^^^^^^^^^^^
		     O             A             O
		     
		^^^^^^^^^^^^^            ^^^^^^^^^^^^^^
		     B             O             C
		
		In such a configuration, would be counted, in order:
		Overlap O/B (1)
		Overlap O/B (2)
		Overlap A/B (3)
		Overlap A/O (4)
		Overlap A/O (5)
		Overlap A/C (6)
		Overlap O/C (7)
		Overlap O/C (8)
		'''
		if config["remove_IOB"]:
			df['y_value']	=	df['y_value'].apply(lambda x: x[2:] if x != 'O' else 'NONE')
			df['y_predict']	=	df['y_predict'].apply(lambda x: x[2:] if x != 'O' else 'NONE')

		df_ct	=	pd.crosstab(df['y_value'],df['y_predict'],rownames=['y_value'],colnames=['y_predict'])
	
	########
	# Improving the dataframe before display
	########
	df_ct.fillna(0.0,inplace=True)
	# df_ct.reindex(sorted(df_ct.columns))
	df_ct			=	df_ct.reindex(sorted(df_ct.columns),axis=1)
	df_ct.index.name	=	'y_predict'
	df_ct.columns.name	=	'y_value'
	# df_ct.sort_values(df_ct.index,inplace=True)
	df_ct.sort_values(df_ct.index.name,inplace=True)
	print(df_ct)

	# These are the scaling options, I tested them and "symlog" seems to be the most interesting for display
	# >>> matplotlib.scale.get_scale_names()
	# ['asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog']

	plt.matshow(df_ct,cmap=plt.cm.Blues,norm='symlog')

	plt.xlabel('y_predict')
	plt.ylabel('y_value')
	plt.xticks(np.arange(len(df_ct.columns)), df_ct.columns, rotation=90)
	plt.yticks(np.arange(len(df_ct.columns)), df_ct.columns)
	plt.colorbar()

	plt.show()

	########
	# In preparation of finding overlapping sentences, determine the most overlapping classes
	########
	max_cell_value	=	-1
	max_x		=	None
	max_y		=	None

	for i in df_ct.columns:
		if config["ignore_default_for_overlapping_classes"] and i in ['O','NONE']:
			continue
		for j in df_ct.columns:
			if i == j:
				continue
			if config["ignore_default_for_overlapping_classes"] and j in ['O','NONE']:
				continue
			local_value	=	df_ct.loc[i,j]
			if local_value > max_cell_value:
				max_cell_value	=	local_value
				max_x		=	j
				max_y		=	i
	if config["confusion_matrix_based_on_full_annotations"]:
		print(f'{"-"*64}\n{max_cell_value} annotations de {max_y} ont été annotées (au moins partiellement) par {max_x} (MAXIMUM)\n{"-"*64}')
	else:
		print(f'{"-"*64}\n{max_cell_value} tokens individuels de {max_y} ont été annotées par {max_x} (MAXIMUM)\n{"-"*64}')

	########
	# Display examples, with indications of text_id, offsets, etc.
	########
	current_offset_start		=	-1
	current_offset_end		=	-1
	current_string			=	''
	current_text_id			=	''

	display_count			=	0
	
	"""
	print(f'{"text_id":<32} {"offset_start":<16} {"offset_end":<16} {"text":<16}')
	for index, row in df.iterrows():
		if row['text_id'] != current_text_id:
			if current_string != '':
				print(f'{current_text_id:<32} {current_offset_start:>16} {current_offset_end:>16} {current_string}')
				current_string	=	''
				display_count	+=	1
				if display_count >= config["display_max"]:
					break
			# ADDED TABS HERE
			current_text_id		=	row['text_id']
			current_offset_start	=	-1
			current_offset_end	=	-1
				
		if (row['y_value']		==	max_y)	and \
		   (row['y_predict']		==	max_x)	and \
		   (current_offset_end		==	-1):
			current_string		=	row['X_text']
			current_offset_start	=	row['offset_start']
			current_offset_end	=	row['offset_end']	
		elif (row['y_value']		==	max_y)	and \
			 (current_offset_end	!=	-1):
			 
			offset_difference	=	row['offset_start'] - current_offset_end
			current_string		+=	" " * offset_difference
			current_string		+=	row['X_text']
			current_offset_end	=	row['offset_end']
		else:
			if current_string != '':
				print(f'{current_text_id:<32} {current_offset_start:>16} {current_offset_end:>16} {current_string}')
				current_string	=	''
				display_count	+=	1
				if display_count >= config["display_max"]:
					break
				# ADDED TABS HERE
				current_offset_start	=	-1
				current_offset_end	=	-1
	"""
	
	
	
	print(f'{"text_id":<32} {"offsets y_value":<20} {"offsets y_predict":<20} {"text":<16}')
	
	df['next_text_id']	=	df['text_id'].shift(-1)
	df['next_text_id'].fillna('NONE',inplace=True)
	
	series_y_values		=	[]
	series_y_predict	=	[]
	previous_y_value	=	'NONE'
	previous_y_predict	=	'NONE'
	for index, row in df.iterrows():
		########
		# Check if it's the y_value we're looking for
		########
		if row['y_value'] == max_y:
			if row['y_value'] != previous_y_value:
				series_y_values.append([])
			series_y_values[-1].append({
				'X_text'	:	row['X_text'],
				'offset_start'	:	row['offset_start'],
				'offset_end'	:	row['offset_end']
			})
		previous_y_value	=	row['y_value']
		
		########
		# Check if it's the y_predict we're looking for
		########
		if row['y_predict'] == max_x:
			if row['y_predict'] != previous_y_predict:
				series_y_predict.append([])
			series_y_predict[-1].append({
				'X_text'	:	row['X_text'],
				'offset_start'	:	row['offset_start'],
				'offset_end'	:	row['offset_end']
			})
		previous_y_predict	=	row['y_predict']
		
		########
		# Check if we've reached the last row of a text; if so, calculate and display all the overlaps based on series_y_values and series_y_predict
		########
		if row['text_id'] != row['next_text_id']:
			for i in series_y_values:
				y_values_min	=	i[0]['offset_start']
				y_values_max	=	i[-1]['offset_end']
				for j in series_y_predict:
					y_predict_min	=	j[0]['offset_start']
					y_predict_max	=	j[-1]['offset_end']
					if (y_values_min <= y_predict_min and y_predict_min <= y_values_max) or \
						(y_values_min <= y_predict_max and y_predict_max <= y_values_max):
						local_tokens		=	[]
						y_values_offsets	=	[]
						y_predict_offsets	=	[]
						for k in i:
							if k not in local_tokens:
								local_tokens.append(k)
							y_values_offsets.append((k['offset_start'],k['offset_end']))
						for k in j:
							if k not in local_tokens:
								local_tokens.append(k)
							y_predict_offsets.append((k['offset_start'],k['offset_end']))
						
						local_tokens.sort(key=lambda x:x['offset_start'])
						
						display_str	=	''
						for k in local_tokens:
							display_str	=	f'{display_str}{" "*((k["offset_start"]-min(y_values_min,y_predict_min))-len(display_str))}{k["X_text"]}'
						
						print(f'{row["text_id"]:<32} {str(y_values_min)+"-"+str(y_values_max):<20} {str(y_predict_min)+"-"+str(y_predict_max):<20} {display_str:<16}')
						
						y_values_offsets.sort(key=lambda x:x[0])
						display_str	=	''
						for k in y_values_offsets:
							display_str	=	f'{display_str}{" "*(k[0]-min(y_values_min,y_predict_min)-len(display_str))}{"^"*(k[1]-k[0])}'
						display_str	=	f'{max_y:>74} {display_str}'
						print(display_str)
						y_predict_offsets.sort(key=lambda x:x[0])
						display_str	=	''
						for k in y_predict_offsets:
							display_str	=	f'{display_str}{" "*(k[0]-min(y_values_min,y_predict_min)-len(display_str))}{"^"*(k[1]-k[0])}'
						display_str	=	f'{max_x:>74} {display_str}'
						print(display_str)
						print("-"*74)
						display_count	+=	1
					if display_count >= config["display_max"]:
						break
				if display_count >= config["display_max"]:
					break
			series_y_values		=	[]
			series_y_predict	=	[]
			previous_y_value	=	'NONE'
			previous_y_predict	=	'NONE'
			current_text_id		=	row['text_id']
		
		if display_count >= config["display_max"]:
			break
	return 0

if __name__ == '__main__':
	main_result =   main()
	print(f'Program ended with status {main_result}')
