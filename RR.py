import pandas as pd
from scipy import spatial
import re


df_test = pd.read_csv('df_test.csv',sep='\t',encoding='UTF-8')
f_out = open('final_df.csv','w',encoding='UTF-8')
f_out.write('TEXT\tVECTOR\tLABEL\tPRED_LABEL\n')


class_mean = {
'PREAMBLE'          :   -0.008957748123343136,
'FAC'               :   -0.008650351889978716,
'RLC'               :   -0.008840211467011957,
'ISSUE'             :   -0.009236995210093293,
'ARG_PETITIONER'    :   -0.008719417135022124,
'ARG_RESPONDENT'    :   -0.008655683176690026,
'ANALYSIS'	        :   -0.008731448519037304,
'STA'               :   -0.009405967364666843,
'PRE_RELIED'	    :   -0.008873227407148786,
'PRE_NOT_RELIED'    :   -0.008721100965574864,
'RATIO'	            :   -0.009049526857796,
'RPC'               :   -0.009334025454064284,
'NONE'	            :   -0.00907588806659193
}

###############################
#   PREDICTION DES CLASSES    #
###############################


def get_keys_from_value(class_mean, val):
    return [k for k, v in class_mean.items() if v == val]

for index, row in df_test.iterrows():
    vect = row['VEC']
    label = row['LABEL']
    text = row['SENTENCE']
    min_dist = ((min(class_mean.values(), key=lambda x:abs(x-vect))))
    prediction_label = get_keys_from_value(class_mean, min_dist)

    f_out.write(f'{text}\t{vect}\t{label}\t{prediction_label}\n')




###############################
#   CALCUL PRECISION RAPPEL   #
###############################

df = pd.read_csv('final_df.csv',sep='\t',encoding='UTF-8')

ref_pos = ['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE']


for j in ref_pos:

    print(f'------------------{j}------------------')


    ############### PRECISION ###############
    VP = 0
    FP = 0
    for index, row in df.iterrows():

        row['PRED_LABEL'] = re.sub("'\]",'',row['PRED_LABEL'])
        row['PRED_LABEL'] = re.sub("\['",'',row['PRED_LABEL'])

        if (row['LABEL'] == row['PRED_LABEL']):
            if row['LABEL'] == j:
                VP += 1
        else:
            if row['LABEL'] != j and row['PRED_LABEL'] == j:
                FP += 1
    print('PRECISION')
    try:            
        precision = (VP/(VP+FP))
        print(precision)
    except ZeroDivisionError:
        pass
    


    ############### RAPPEL ###############
    VP = 0
    FN = 0
    for index, row in df.iterrows():
        if (row['LABEL'] == row['PRED_LABEL']):
            if row['LABEL'] == j:
                VP += 1
        else:
            if row['LABEL'] == j and row['PRED_LABEL'] != j:
                FN += 1
    print('RAPPEL')
    try:
        rappel = (VP/(VP+FN))
    except ZeroDivisionError:
        pass
    print(rappel)


    ############### F1 SCORE ###############
    print('F1 SCORE')
    try:
        print(2*((rappel*precision)/(rappel+precision)))
    except:
        pass












