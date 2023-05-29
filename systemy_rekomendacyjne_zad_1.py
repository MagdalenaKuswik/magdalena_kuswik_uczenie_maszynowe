import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('GroceryStoreDataSet.csv', header=None, names=['Products'])

df.replace(to_replace='COCK', value='COKE', regex=True, inplace=True)
df.replace(to_replace='SUGER', value='SUGAR', regex=True, inplace=True)

data = list(df["Products"].apply(lambda x:x.split(',')))

TrueFalse = TransactionEncoder()
TrueFalse_data = TrueFalse.fit(data).transform(data)
df2 = pd.DataFrame(TrueFalse_data,columns=TrueFalse.columns_)

df3 = apriori(df2, min_support=0.01, use_colnames=True)
df_ar = association_rules(df3, metric="confidence", min_threshold=0.06)
df_ar = df_ar.sort_values(by='lift', ascending=False)
print(df_ar.head(10))
df_ar = df_ar.sort_values(by='confidence', ascending=False)
print(df_ar.head(10))