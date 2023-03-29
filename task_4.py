from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv('cleaned_data.csv')
print(len(df.columns) - 1)

pd.DataFrame(abs(df.corr()['Credit_Score'].drop('Credit_Score')*100).sort_values(ascending=False)).plot.bar(figsize = (10, 8))

exclude_filter = ~df.columns.isin(['Unnamed: 0', 'Credit_Score'])
pca = PCA().fit(df.loc[:, exclude_filter])

pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(df.loc[:, exclude_filter])
principal_df = pd.DataFrame(data=principal_components)
print(principal_df.head())

print(pca.fit_transform(principal_df))

x_train, x_test, y_train, y_test = train_test_split(principal_df, df['Credit_Score'], test_size=0.33, random_state=42)
print(x_train, x_test, y_train, y_test)

clf = LogisticRegression(random_state=100)
fit = clf.fit(x_train, y_train)
predict = clf.predict(x_test)

matrix = confusion_matrix(y_test, predict)
accuracy = matrix.diagonal().sum() / matrix.sum()
recall = matrix.diagonal() / matrix.sum(axis=1)
precision = matrix.diagonal() / matrix.sum(axis=0)
print('Accuracy: ', accuracy, 'Recall: ', recall, 'Precision: ', precision)
ConfusionMatrixDisplay(matrix).plot()
plt.show()

print(classification_report(y_test, predict))