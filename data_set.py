import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('NER_TRAIN_PREAMBLE.csv',encoding='UTF-8',sep='\t')
data = df['annotation_text']
label = df['annotation_label']

# random_state=None
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=None)


print('------------\nTRAIN\n------------')
print(f"COURT\t{y_train.value_counts()['COURT']}")
print(f"JUDGE\t{y_train.value_counts()['JUDGE']}")
print(f"LAWYER\t{y_train.value_counts()['LAWYER']}")
print(f"PETITIONER\t{y_train.value_counts()['PETITIONER']}")
print(f"RESPONDANT\t{y_train.value_counts()['RESPONDENT']}")

print('------------\nDEV\n------------')
print(f"COURT\t{y_test.value_counts()['COURT']}")
print(f"JUDGE\t{y_test.value_counts()['JUDGE']}")
print(f"LAWYER\t{y_test.value_counts()['LAWYER']}")
print(f"PETITIONER\t{y_test.value_counts()['PETITIONER']}")
print(f"RESPONDANT\t{y_test.value_counts()['RESPONDENT']}")
