import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('NER_TRAIN_JUDGEMENT.csv',encoding='UTF-8',sep='\t')

data = df['annotation_text']
label = df['annotation_label']

# random_state=None : Use the global random state instance from numpy.random. Calling the function multiple times will reuse the same instance, and will produce different results.
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=None)


print('------------\nTRAIN\n------------')
print(f"COURT\t{y_train.value_counts()['COURT']}")
print(f"JUDGE\t{y_train.value_counts()['JUDGE']}")
print(f"PETITIONER\t{y_train.value_counts()['PETITIONER']}")
print(f"RESPONDANT\t{y_train.value_counts()['RESPONDENT']}")
print(f"DATE\t{y_train.value_counts()['DATE']}")
print(f"ORG\t{y_train.value_counts()['ORG']}")
print(f"GPE\t{y_train.value_counts()['GPE']}")
print(f"STATUTE\t{y_train.value_counts()['STATUTE']}")
print(f"PROVISION\t{y_train.value_counts()['PROVISION']}")
print(f"PROVISION\t{y_train.value_counts()['PROVISION']}")
print(f"PRECEDENT\t{y_train.value_counts()['PRECEDENT']}")
print(f"CASE_NUMBER\t{y_train.value_counts()['CASE_NUMBER']}")
print(f"WITNESS\t{y_train.value_counts()['WITNESS']}")
print(f"OTHER_PERSON\t{y_train.value_counts()['OTHER_PERSON']}")



print('------------\nDEV\n------------')
print(f"COURT\t{y_test.value_counts()['COURT']}")
print(f"JUDGE\t{y_test.value_counts()['JUDGE']}")
print(f"PETITIONER\t{y_test.value_counts()['PETITIONER']}")
print(f"RESPONDANT\t{y_test.value_counts()['RESPONDENT']}")
print(f"DATE\t{y_train.value_counts()['DATE']}")
print(f"ORG\t{y_train.value_counts()['ORG']}")
print(f"GPE\t{y_train.value_counts()['GPE']}")
print(f"STATUTE\t{y_train.value_counts()['STATUTE']}")
print(f"PROVISION\t{y_train.value_counts()['PROVISION']}")
print(f"PROVISION\t{y_train.value_counts()['PROVISION']}")
print(f"PRECEDENT\t{y_train.value_counts()['PRECEDENT']}")
print(f"CASE_NUMBER\t{y_train.value_counts()['CASE_NUMBER']}")
print(f"WITNESS\t{y_train.value_counts()['WITNESS']}")
print(f"OTHER_PERSON\t{y_train.value_counts()['OTHER_PERSON']}")
