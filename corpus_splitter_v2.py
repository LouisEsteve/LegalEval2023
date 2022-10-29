import pandas as pd
from math import floor

text_base_path	=	'data/text_base.csv'
encoding	=	'UTF-8'
sep		=	'\t'

train_size	=	0.8

df		=	pd.read_csv(text_base_path,encoding=encoding,sep=sep)

text_id_list	=	df['id']

unique_id_list	=	[]

for i in text_id_list:
	if i not in unique_id_list:
		unique_id_list.append(i)


index_limit	=	floor(len(unique_id_list)*0.8)

train_ids	=	pd.Series(unique_id_list[:index_limit])
train_ids.name	=	'id'
dev_ids		=	pd.Series(unique_id_list[index_limit:])
dev_ids.name	=	'id'

train		=	df.merge(train_ids,left_on='id',right_on='id')
dev		=	df.merge(dev_ids,left_on='id',right_on='id')

train.to_csv('CUSTOM_NER_TRAIN_TEXT_BASE.csv',encoding=encoding,sep=sep)
dev.to_csv('CUSTOM_NER_DEV_TEXT_BASE.csv',encoding=encoding,sep=sep)
print('Wrote TRAIN and DEV text base files')