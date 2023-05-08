import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Salary Data.csv')
df.reset_index()
df.head()
df.shape

df.isna().sum()

df.dropna(axis=0, inplace=True)

df.isna().sum()

X = df['Years of Experience'].values.reshape(-1,1)
y = df['Salary'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

'''
print(f"\nModel score: {model.score(X_test, y_test)}" )
plt.scatter(X_train, y_train, c='g')
plt.plot(np.linspace(0,25,100).reshape(-1,1), model.predict(np.linspace(0,25,100).reshape(-1,1)), 'm')
plt.title('Training set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, y_test, c='g')
plt.plot(np.linspace(0,25,100).reshape(-1,1), model.predict(np.linspace(0,25,100).reshape(-1,1)), 'm')
plt.title('Test set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
'''
#print(model.predict(np.array([2.34]).reshape(-1,1)))
#print(model.predict(np.array([0.5]).reshape(-1,1)))
#Zmiana Gender  female: 0 male: 1
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

#Lasso
#Wybór zmiennych objaśniających i zmiennej objaśnianej:
X = df[['Age', 'Gender', 'Years of Experience']]
y = df['Salary']

#Podział danych na zbiór treningowy i testowy:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Budowanie modelu LASSO
lasso = Lasso(alpha=0.1)

#Trenowanie modelu na danych treningowych:
lasso.fit(X_train, y_train)

#Wykonanie predykcji na danych testowych:
y_pred = lasso.predict(X_test)

#Obliczenie wartości MSE i R²:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Lasso: ')
print('MSE: ', mse)
print('R²: ', r2 , '\n')

#Tworzenie cech wielomianowych stopnia 3:
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

#Podział danych na zbiór treningowy i testowy:
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

#Budowanie modelu regresji wielomianowej
#Definiowanie modelu:

poly_reg = LinearRegression()

#Trenowanie modelu na danych treningowych:

poly_reg.fit(X_train, y_train)

#Ocena modelu
#Wykonanie predykcji na danych testowych:

y_pred = poly_reg.predict(X_test)

#Obliczenie wartości MSE i R²:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Regresja wielomianowa 3 stopnia: ')
print('MSE: ', mse)
print('R²: ', r2, '\n')

#Budowanie modelu regresji drzewa decyzyjnego

#Definiowanie modelu i ustawienie parametru max_depth:
dt_reg = DecisionTreeRegressor(max_depth=5)
#Trenowanie modelu na danych treningowych:
dt_reg.fit(X_train, y_train)
#Wykonanie predykcji na danych testowych:
y_pred_dt = dt_reg.predict(X_test)
#Obliczenie wartości MSE i R²:
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print('Regresja drzewa decyzyjnego: ')
print('MSE: ', mse_dt)
print('R²: ', r2_dt)

# utwórz listę z danymi dla każdego modelu
data = [['Lasso', 218083012.19, 0.892693051],
        ['Regresja wielomianowa 3st', 229245542.26, 0.887200569],
        ['Regresja drzewa decyzyj.', 240249514.07, 0.881786105]]

# utwórz obiekt DataFrame z danymi i nazwami kolumn
df = pd.DataFrame(data, columns=['Model', 'MSE', 'R²'])

# wyświetl tabelę
print(df)
