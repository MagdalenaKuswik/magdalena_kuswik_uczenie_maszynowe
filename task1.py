import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#zaczytaj dane z pliku csv

df_train = pd.read_csv("train.csv" , sep = "," , encoding= 'utf-8')
print(df_train)
#sprawdź liczbę kolumn i wierszy
df_train
#wyświetl część tabeli
df_train.head()
#usuń wiersze z duplikatami id
df_train.drop_duplicates(subset="ID", inplace=True)
#opisz statystyki danych
df_train.describe()
#zlicz różne wartości danych
for i in df_train.columns: print(df_train[i].value_counts())
print('*'*50)
df_train.info()
#zmień dane na numeryczne
FeaturesToConvert = ['Age', 'Annual_Income',
'Num_of_Loan', 'Num_of_Delayed_Payment',
'Changed_Credit_Limit', 'Outstanding_Debt',
'Amount_invested_monthly', 'Monthly_Balance' ]
# ale najpierw sprawdź czy nie ma błędów w danych
for feature in FeaturesToConvert:
    uniques = df_train[feature].unique()
print('Feature:', '\n', '\n', uniques, '\n', '--'*40, '\n')
print(df_train)
# usuń zbędne znaki '-’ , '_'
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].str.strip('-_')
# puste kolumny zastąp NAN
for feature in FeaturesToConvert: df_train[feature] = df_train[feature].replace({'':np.nan}) # zmien typ zmiennych ilościowych for feature in FeaturesToConvert:
df_train[feature] = df_train[feature].astype('float64')
#uzupełnij braki średnią
df_train['Monthly_Inhand_Salary']= df_train['Monthly_Inhand_Salary'].fillna(method='pad')

# stwórz obiekt enkodera
le = LabelEncoder()
df_train.Occupation = le.fit_transform(df_train.Occupation)
# sprawdź transformacje
cols = ['Month', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
# Encode labels of multiple columns at once df[cols] =
df_train[cols].apply(LabelEncoder().fit_transform)
# Print head
print(df_train.head())
print(df_train.columns)
