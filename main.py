import re
import time
import pandas as pd
import json

#######################################

encoding	=	'UTF-8'

data_path	=	'data'

"""
data_dict	=	{
					'NER_TRAIN_PREAMBLE'	:	{
						'csv_path'			:	f'{data_path}/NER_TRAIN_PREAMBLE.csv',
						'regex_patterns'	:	{
							'COURT'	:	[
								# '[hH]igh\\s[cC]ourt\\s[oO]f\\s[A-Z].*?\\b',
								'([hH]igh|[sS]upreme)\\s[cC]ourt\\s[oO]f\\s[A-Z].*?\\b[aA]t( [A-Z].*?\\b)+',
								# '([hH]igh|[sS]upreme)\\s[cC]ourt\\s[oO]f\\s[A-Z]\\S*(\\s[aA]t( [A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s[cC]ourt\\s[oO]f( [A-Z]\\S*)+( [aA]t( [A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Z][^\\n \\t\\r,\\.]*)+(\\s+[aA]t(\\s+[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Z][^\\n \\t]*)+(\\s+[aA]t(\\s+[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Z].*?\\b)+(\\s+[aA]t([ \\t]+[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Z].*?\\b)+(\\s+[aA]t\\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Z].*?\\b)+(\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Za-z].*?\\b)+(\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '(([hH]igh|[sS]upreme)\\s+)?[cC]ourt\\s+[oO]f( [A-Za-z].*?\\b)+(,?\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f( [A-Za-z].*?\\b)+(,?\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z].*?\\b)|\\s?[:;])+(,?\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z][^\\n ]*\\b)|\\s?[:;])+(,?\\s+[aA]t\n?([ \\t]*[A-Z].*?\\b)+)?',	#0.84, 0.59
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z][^\\n ]*\\b)|\\s?[:;])+(,?\\s+[aA]t\n?(,?[ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z][^\\n ]*\\b)|\\s?[:;\\.])+(,?\\s+[aA]t\n?(,?[ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z][^\\n ]*\\b)|\\s?[:;\\.])+,?(\\s+[aA]t\n?(,?[ \\t]*[A-Z].*?\\b)+)?',
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]),?)+(\\s+[aA]t\n?(,?[ \\t]*[A-Z].*?\\b)+)?',	#0.85, 0.59
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(,?( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]))+(\\s+[aA]t\n?(,?[ \\t]*[A-Z].*?\\b)+)?',	#0.86, 0.60
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(,?( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]))+(\\s+[aA]t\\n?(,?[ \\t]*[A-Z].*?\\b)+)?',	#0.86, 0.60
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(,?( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]))+(\\s+[aA]t\\n?[^\\n]+)?',	#0.86, 0.60
								# '([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(,?( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]))+(\\s+[aA]t\\n{0,1}[^\\n]+)?',	#0.86, 0.60
								'([A-Z].*?\\b [bB]ench [oO]f [A-Z].*?\\b )?([hH]igh|[sS]upreme)\\s+[cC]ourt\\s+[oO]f(,?( [A-Za-z][^\\n ]*\\b|\\s?[:;\\.]))+(\\s+[aA]t\\n{0,1}[^\\n]+)?',	#0.86, 0.60
								# '(?<=[iI]n [tT]he )([A-Z]\\S* )*[cC]ourts?( [A-Z][^\\n ]*)*',
								'(?<=[iI]n [tT]he )([A-Z]\\S* )*([cC]ourt|[tT]ribunal|[sS]ession)s?( [A-Z][^\n ]*)*',	#0.61, 0.52
								# '([A-Z].*?\\b [bB]ench [oO]f [A-Z].*?\\b )?([hH]igh|[sS]upreme)\\s+([cC]ourt|[tT]ribunal|[sS]ession)s?( [A-Z][^\n ]*)*',	#0.70, 0.56
								# '([A-Z].*?\\b [bB]ench [oO]f [A-Z].*?\\b )?(([hH]igh|[sS]upreme)\\s+)?([cC]ourt|[tT]ribunal|[sS]ession)s?( [A-Z][^\n ]*)*(,?\\n[^\\n]+)?',	#0.29, 0.50
							]
						}
					}
				}
"""

regex_config_path	=	'regex_config.json'

regex_config_file	=	open(regex_config_path, 'rt', encoding=encoding)

data_dict		=	json.load(regex_config_file)

print_false_positives	=	False

#######################################

def main() -> int:
	text_base_path		=	f'{data_path}/text_base.csv'
	text_base_file		=	open(text_base_path, 'rt', encoding=encoding)
	text_base_df		=	pd.read_csv(text_base_file, sep='\t')
	text_base_df['text']	=	text_base_df['text'].str.replace('\\n','\n',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\t','\t',regex=False)
	text_base_df['text']	=	text_base_df['text'].str.replace('\\r','\r',regex=False)
	print(text_base_df)
	print(f'Loaded {text_base_path}')
	
	###################################################
	
	for i in data_dict:
		print(f'Starting {i} with CSV from path {data_dict[i]["csv_path"]}')
		
		set_file	=	open(data_dict[i]['csv_path'], 'rt', encoding=encoding)
		set_df		=	pd.read_csv(set_file, sep='\t')
		
		for j in data_dict[i]['regex_patterns']:
			print(f'Starting {j}')
			
			for k in data_dict[i]['regex_patterns'][j]:
				# print(f'Compiling {k}')
				local_regex		=	re.compile(k)
				
				count_true_positive	=	0
				count_false_positive	=	0
				count_false_negative	=	0
				
				for m in text_base_df['id']:
					if set_df[set_df['text_id'] == m].shape[0] == 0:
						continue
					
					local_text			=	text_base_df['text'][text_base_df.index[text_base_df['id'] == m].tolist()[0]]
					
					local_count_true_positive	=	0
					local_count_false_positive	=	0
					local_count_false_negative	=	0
				
					regex_result			=	local_regex.finditer(local_text)
					
					expected_results		=	set_df.loc[(set_df['annotation_label'] == j) & (set_df['text_id'] == m)]
					
					for n in regex_result:
						start	=	n.start()
						end	=	n.end()
						# print(n)
						
						if expected_results.loc[(expected_results['annotation_start'] == start) & (expected_results['annotation_end'] == end)].shape[0] > 0:
							local_count_true_positive	+=	1
						else:
							if print_false_positives:
								print(f'False positive: {n}')
							local_count_false_positive	+=	1
					
					local_count_false_negative	=	expected_results.shape[0] - local_count_true_positive
					
					count_true_positive		+=	local_count_true_positive
					count_false_positive		+=	local_count_false_positive
					count_false_negative		+=	local_count_false_negative
				
				# print(f'{k}\t{count_true_positive}\t{count_false_positive}\t{count_false_negative}')
				print(f'{k}\t{count_true_positive/(count_true_positive+count_false_positive)}\t{count_true_positive/(count_true_positive+count_false_negative)}')
		
		set_file.close()
	
	###################################################
	
	text_base_file.close()
	return 0

if __name__ ==	'__main__':
	t	=	time.time()
	result	=	main()
	print(f'Program ended with status {str(result)} in {str(time.time()-t)}s')

regex_config_file.close()
