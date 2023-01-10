import pandas as pd
import json

config_path	=	'TRAIN_DEV_MERGE_config.json'

try:
	config_file	=	open(config_path,'rt',encoding='UTF-8')
	config		=	json.load(config_file)
	config_file.close()
except IOError or OSError as e:
	print(e)
	print(f'Could not open {config_path}. Exiting.')
	exit()

###

def merge(in1,in2,out,update_id):
	df1	=	pd.read_csv(in1,sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
	df2	=	pd.read_csv(in2,sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
	if update_id:
		df2.index	+=	len(df1) #!
	df3	=	pd.concat([df1,df2])
	df3.to_csv(out,sep=config["SEP"])
	print(f'Wrote {out}')

merge(config["TEXT_BASE_TRAIN_PATH_IN"],config["TEXT_BASE_DEV_PATH_IN"],config["TEXT_BASE_PATH_OUT"],False)
merge(config["ANNOT_PREAMBLE_TRAIN_PATH_IN"],config["ANNOT_PREAMBLE_DEV_PATH_IN"],config["ANNOT_PREAMBLE_PATH_OUT"],True)
merge(config["ANNOT_JUDGEMENT_TRAIN_PATH_IN"],config["ANNOT_JUDGEMENT_DEV_PATH_IN"],config["ANNOT_JUDGEMENT_PATH_OUT"],True)
merge(config["FEATURES_TRAIN_PATH_IN"],config["FEATURES_DEV_PATH_IN"],config["FEATURES_PATH_OUT"],True)

'''
df1	=	pd.read_csv(config["TEXT_BASE_TRAIN_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(config["TEXT_BASE_DEV_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df3	=	pd.concat([df1,df2])
df3.to_csv(config["TEXT_BASE_PATH_OUT"],sep=config["SEP"])
print(f'Wrote {config["TEXT_BASE_PATH_OUT"]}')

df1	=	pd.read_csv(config["ANNOT_PREAMBLE_TRAIN_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(config["ANNOT_PREAMBLE_DEV_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2.index	+=	len(df1) #!
df3	=	pd.concat([df1,df2])
df3.to_csv(config["ANNOT_PREAMBLE_PATH_OUT"],sep=config["SEP"])
print(f'Wrote {config["ANNOT_PREAMBLE_PATH_OUT"]}')

df1	=	pd.read_csv(config["ANNOT_JUDGEMENT_TRAIN_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(config["ANNOT_JUDGEMENT_DEV_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2.index	+=	len(df1) #!
df3	=	pd.concat([df1,df2])
df3.to_csv(config["ANNOT_JUDGEMENT_PATH_OUT"],sep=config["SEP"])
print(f'Wrote {config["ANNOT_JUDGEMENT_PATH_OUT"]}')

df1	=	pd.read_csv(config["FEATURES_TRAIN_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(config["FEATURES_DEV_PATH_IN"],sep=config["SEP"],index_col=["id"],dtype={"text_id":"str"})
df2.index	+=	len(df1) #!
df3	=	pd.concat([df1,df2])
df3.to_csv(config["FEATURES_PATH_OUT"],sep=config["SEP"])
print(f'Wrote {config["FEATURES_PATH_OUT"]}')
'''