import json
from os import listdir
from os.path import exists
import re
import pandas as pd

# data_path			=	'data'
data_path			=	'NER_DEV/NER_DEV'

encoding			=	'UTF-8'

sep				=	'\t'
csv_replacements		=	[
	['\t', '\\t'],
	['\n', '\\n'],
	['\r', '\\r'],
	['"', '\\"'],
	["'", "\\'"]
]

tab				=	'\t'
tab_replacement			=	'\\t'
linefeed			=	'\n'
linefeed_replacement		=	'\\n'
carriage_return			=	'\r'
carriage_return_replacement	=	'\\r'

text_dict			=	{}

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
	files	=	listdir(data_path)
	for i in files:
		if not i.endswith('.json'):
			continue
		local_path = f'{data_path}/{i}'
		print(local_path)
		try:
			local_file = open(local_path, 'rt', encoding='utf-8')
		except IOError or OSError as m:
			print(m)
		else:
			sum_for_average			=	0
			count_for_average		=	0
			
			json_object			=	json.load(local_file)
			#print(dir(json_object))

		
			list_text_id			=	[]
			list_text_group			=	[]
			list_text_source		=	[]
			list_annotation_result_group	=	[]
			list_annotation_id		=	[]
			list_annotation_type		=	[]
			list_annotation_to_name		=	[]
			list_annotation_from_name	=	[]
			list_annotation_start		=	[]
			list_annotation_end		=	[]
			list_annotation_text		=	[]
			list_annotation_label		=	[]

			j_count = 1
			for j in json_object:
				print(f'Parsing text {j_count}/{len(json_object)}...', end='\r')
				
				text_id			=	j['id']			if j['id'] else hash(j['data']['text'])			#j.id
				
				if text_id not in text_dict:
					text_dict[text_id]	=	j['data']['text']
				
				sum_for_average		+=	len(j['data']['text'])
				count_for_average	+=	1
				
				text_group		=	j['meta']['group']	if 'group' in j['meta'].keys() else '_'			#j.meta.group
				text_source		=	j['meta']['source']	if 'source' in j['meta'].keys() else '_'		#j.meta.group

				for q in range(len(j['annotations'])):
					annotation_result_group	 =	q
					
					for k in j['annotations'][q]['result']:
						annotation_id		=	k['id']								#k.id
						annotation_type		=	k['type']		if 'type' in k.keys() else '_'		#k.type
						annotation_to_name	=	k['to_name']		if 'to_name' in k.keys() else '_'	#k.to_name
						annotation_from_name	=	k['from_name']		if 'from_name' in k.keys() else '_'	#k.from_name
						annotation_start	=	k['value']['start']						#k.value.start
						annotation_end		=	k['value']['end']						#k.value.end
						annotation_text		=	k['value']['text']						#k.value.text

						for m in k['value']['labels']:
							annotation_label	=	m
							
							list_text_id.append(text_id)
							list_text_group.append(text_group)
							list_text_source.append(text_source)
							list_annotation_result_group.append(annotation_result_group)
							list_annotation_id.append(annotation_id)
							list_annotation_type.append(annotation_type)
							list_annotation_to_name.append(annotation_to_name)
							list_annotation_from_name.append(annotation_from_name)
							list_annotation_start.append(annotation_start)
							list_annotation_end.append(annotation_end)
							list_annotation_text.append(annotation_text)
							list_annotation_label.append(annotation_label)
				
				
				j_count += 1
			local_file.close()
			print('')
			
			output_count = 0
			output_path = f'{local_path[:-5]}_restructured{str(output_count)}.csv'
			while(exists(output_path)):
				output_count += 1
				output_path = f'{local_path[:-5]}_restructured{str(output_count)}.csv'
			
			df				=	pd.DataFrame()
			
			df.index.name			=	'id'
			df['text_id']			=	pd.Series(list_text_id)
			df['text_group']		=	pd.Series(list_text_group)
			df['text_source']		=	pd.Series(list_text_source)
			df['annotation_result_group']	=	pd.Series(list_annotation_result_group)
			df['annotation_id']		=	pd.Series(list_annotation_id)
			df['annotation_type']		=	pd.Series(list_text_id)
			df['annotation_to_name']	=	pd.Series(list_annotation_type)
			df['annotation_from_name']	=	pd.Series(list_annotation_from_name)
			df['annotation_start']		=	pd.Series(list_annotation_start)
			df['annotation_end']		=	pd.Series(list_annotation_end)
			df['annotation_text']		=	pd.Series(list_annotation_text)
			df['annotation_label']		=	pd.Series(list_annotation_label)
			
			
			df.to_csv(output_path,sep=sep,encoding='UTF-8')
			
			print(f'Wrote {output_path}')
	text_base_path  =	f'{data_path}/text_base.csv'
	
	list_ids	=	[i for i in text_dict]
	list_texts	=	[text_dict[i] for i in text_dict]
	
	text_df		=	pd.DataFrame()
	text_df['id']	=	pd.Series(list_ids)
	text_df['text']	=	pd.Series(list_texts)
	
	text_df.to_csv(text_base_path,sep=sep,encoding=encoding,index=False)
	print(f'Wrote {text_base_path}')


if __name__ == '__main__':
	main()

