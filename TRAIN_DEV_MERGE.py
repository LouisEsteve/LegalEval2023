import pandas as pd

TEXT_BASE_TRAIN_PATH		=	"data/NER_TRAIN_text_base.csv"
TEXT_BASE_DEV_PATH		=	"NER_DEV/NER_DEV/NER_DEV_text_base.csv"
# ANNOT_PREAMBLE_TRAIN_PATH	=	"data/NER_TRAIN_PREAMBLE.csv"
# ANNOT_JUDGEMENT_TRAIN_PATH	=	"data/NER_TRAIN_JUDGEMENT.csv"
ANNOT_PREAMBLE_TRAIN_PATH	=	"data/NER_TRAIN_PREAMBLE_restructured7.csv"
ANNOT_JUDGEMENT_TRAIN_PATH	=	"data/NER_TRAIN_JUDGEMENT_restructured7.csv"
ANNOT_PREAMBLE_DEV_PATH		=	"NER_DEV/NER_DEV/NER_DEV_PREAMBLE_restructured0.csv"
ANNOT_JUDGEMENT_DEV_PATH	=	"NER_DEV/NER_DEV/NER_DEV_JUDGEMENT_restructured0.csv"
FEATURES_TRAIN_PATH		=	"features_memory_train8.csv"
FEATURES_DEV_PATH		=	"features_memory_dev8.csv"

TEXT_BASE_PATH_OUT		=	"data/NER_MERGED_TEXT_BASE.csv"
ANNOT_PREAMBLE_PATH_OUT		=	"data/NER_MERGED_PREAMBLE.csv"
ANNOT_JUDGEMENT_PATH_OUT	=	"data/NER_MERGED_JUDGEMENT.csv"
FEATURES_PATH_OUT		=	"features_memory_train8_MERGED.csv"


SEP				=	"\t"

###

df1	=	pd.read_csv(TEXT_BASE_TRAIN_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(TEXT_BASE_DEV_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
df3	=	pd.concat([df1,df2])
df3.to_csv(TEXT_BASE_PATH_OUT,sep=SEP)
print(f'Wrote {TEXT_BASE_PATH_OUT}')

df1	=	pd.read_csv(ANNOT_PREAMBLE_TRAIN_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(ANNOT_PREAMBLE_DEV_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
# df1	=	pd.read_csv(ANNOT_PREAMBLE_TRAIN_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
# df2	=	pd.read_csv(ANNOT_PREAMBLE_DEV_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
df2.index	+=	len(df1)
df3	=	pd.concat([df1,df2])
df3.to_csv(ANNOT_PREAMBLE_PATH_OUT,sep=SEP)
print(f'Wrote {ANNOT_PREAMBLE_PATH_OUT}')

df1	=	pd.read_csv(ANNOT_JUDGEMENT_TRAIN_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(ANNOT_JUDGEMENT_DEV_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
# df1	=	pd.read_csv(ANNOT_JUDGEMENT_TRAIN_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
# df2	=	pd.read_csv(ANNOT_JUDGEMENT_DEV_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
df2.index	+=	len(df1)
df3	=	pd.concat([df1,df2])
df3.to_csv(ANNOT_JUDGEMENT_PATH_OUT,sep=SEP)
print(f'Wrote {ANNOT_JUDGEMENT_PATH_OUT}')

df1	=	pd.read_csv(FEATURES_TRAIN_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
df2	=	pd.read_csv(FEATURES_DEV_PATH,sep=SEP,index_col=["id"],dtype={"text_id":"str"})
# df1	=	pd.read_csv(FEATURES_TRAIN_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
# df2	=	pd.read_csv(FEATURES_DEV_PATH,sep=SEP,index_col=None,dtype={"text_id":"str"})
df2.index	+=	len(df1)
df3	=	pd.concat([df1,df2])
df3.to_csv(FEATURES_PATH_OUT,sep=SEP)
print(f'Wrote {FEATURES_PATH_OUT}')