import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('u.data.csv', header=None, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
data = data.drop('timestamp', axis=1)

df = pd.DataFrame(data)

unique_users = data['user_id'].nunique()
unique_movies = data['item_id'].nunique()

print(unique_users)
print(unique_movies)

plt.hist(data['rating'], bins=5, edgecolor='black')
plt.xlabel('Ocena')
plt.ylabel('Liczba')
plt.title('Histogram ocen')
plt.show()