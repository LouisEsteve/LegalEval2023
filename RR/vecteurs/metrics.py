import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sn
import matplotlib.pyplot as plt

df_1 = pd.read_csv('RR_DEV_alt.csv',sep='\t')
df_2 = pd.read_csv('pred.csv',sep='\t')


df = pd.concat([df_1, df_2], axis=1)

confusion_matrix_1 = pd.crosstab(df['annotation_label'], df['label_pred'])
print(confusion_matrix_1)

plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix_1, annot=True, fmt='g',cmap="PiYG")
plt.show()





ref_pos = ['PREAMBLE','FAC','RLC','ISSUE','ARG_PETITIONER','ARG_RESPONDENT','ANALYSIS','STA','PRE_RELIED','PRE_NOT_RELIED','RATIO','RPC','NONE']


for j in ref_pos:

    print(f'--------{j}----------')

    # PRECISION
    VP = 0
    FP = 0

    for index, row in df.iterrows():
        if (row['annotation_label'] == row['label_pred']):
            if row['annotation_label'] == j:
                VP += 1
        else:
            if row['annotation_label'] != j and row['label_pred'] == j:
                FP += 1
    print('PRECISION')  
    try:          
        precision = (VP/(VP+FP))
    except ZeroDivisionError:
        pass
    print(precision)


    # RAPPEL 

    VP = 0
    FN = 0

    for index, row in df.iterrows():
        if (row['annotation_label'] == row['label_pred']):
            if row['annotation_label'] == j:
                VP += 1
        else:
            if row['annotation_label'] == j and row['label_pred'] != j:
                FN += 1

    print('RAPPEL')
    try:
        rappel = (VP/(VP+FN))
    except ZeroDivisionError:
        pass

    print(rappel)


    # F1 SCORE

    print('F1 SCORE')
    try:
        macro = 2*((rappel*precision)/(rappel+precision))
        macro = macro/len(ref_pos)
        print(2*((rappel*precision)/(rappel+precision)))
    except ZeroDivisionError:
        pass

print('\n\n\n')

# ACCURACY
VP = 0
FP = 0
for index, row in df.iterrows():
    if (row['annotation_label'] == row['label_pred']):
        VP += 1
    else:
        FP += 1
print('ACCURACY')        
accuracy = (VP/(VP+FP))
print(accuracy)

print('ACCURACY SKLEARN')
print(accuracy_score(df['annotation_label'], df['label_pred'], normalize=True))


print('\n\n MACRO F1 sklearn')

print(f1_score(df['annotation_label'], df['label_pred'], average='macro'))


print('\n MICRO F1')
print(f1_score(df['annotation_label'], df['label_pred'], average='micro'))