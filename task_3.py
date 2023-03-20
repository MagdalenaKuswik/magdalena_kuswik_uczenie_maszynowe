import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_data.csv", sep = "," , encoding= 'utf-8')
frame = df.drop(['Credit_Score'], axis = 1)
creditScore = df['Credit_Score']

print(frame.head())

print(frame.shape[1])

print(frame.info())

pca = PCA(svd_solver= 'full', n_components= 0.90)
glowne = pca.fit_transform(frame)
glowneFrame = pd.DataFrame(data=glowne)

print(glowneFrame.head())

print(glowneFrame.shape[1])

print(glowneFrame.info())

x_train, x_test, y_train, y_test = train_test_split(glowneFrame, creditScore, test_size= 0.5, random_state=80)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
