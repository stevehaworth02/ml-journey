import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer


df = pd.read_csv("data/Housing.csv")
print(f"Columns: {df.columns}\n Rows: {len(df)}") 
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ('num', Normalizer(), num_cols),
    ('cat', 'passthrough', cat_cols)
])

df = pd.DataFrame(
    preprocessor.fit_transform(df),
    columns=num_cols.union(cat_cols, sort=False)
)
X = df[["area", "bathrooms"]].to_numpy()
y = df["price"].to_numpy()
print(f"Feature Matrix Dimensions: {X.shape}\n Feature Matrix X: {X[0:5]}")

X_train, X_Test, y_train, y_test = train_test_split(X, y, train_size=0.8)