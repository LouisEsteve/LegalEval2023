import pandas as pd
from scipy import spatial
import re
import numpy as np

df = pd.read_csv('RR_TRAIN_alt.csv',sep='\t',encoding='UTF-8')
# df['annotation_label'].replace(['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE'],[0,1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)

"""
t = []
for i in df['annotation_label']:
    t.append(i)

def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = np.zeros((n,n))

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    M = M/M.sum(axis=1, keepdims=True)
    return M

m = transition_matrix(t)
for row in m: print(' '.join(f'{x:.2f}' for x in row))

print(m)
print(type(m))

df_out = pd.DataFrame(m,
columns=['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE']
)
df_out.index = pd.Series(['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE'])
df_out.index.name = 'id'
print(df_out)
"""

transition_probabilities    =   {}
df_gb   =   df.groupby('text_id')
for name, group in df_gb:
    previous_label  =   'PREAMBLE'
    for id, row in group.iterrows():
        if previous_label not in transition_probabilities:
            transition_probabilities[previous_label]    =   {}
        new_label   =   row['annotation_label']
        if new_label not in transition_probabilities[previous_label]:
            transition_probabilities[previous_label][new_label] =   0
        transition_probabilities[previous_label][new_label] +=  1
        previous_label  =   new_label

df_out  =   pd.DataFrame(transition_probabilities)
df_out  =   df_out.fillna(0)
for i in df_out:
    df_out[i]   /=   df_out[i].sum()

df_out.index.name   =   'id'
print(df_out)



df_out.to_csv('RR_matrice_transitions.csv',sep='\t',encoding='UTF-8')