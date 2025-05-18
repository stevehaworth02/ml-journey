# === 1. Import libraries ===
# - numpy
# - matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# === 2. Load & prepare the data ===
df = pd.read_csv("data/Housing.csv")
print(f"Columns: {df.columns}\n Rows: {len(df)}") 
X = df[["area", "bathrooms"]].to_numpy()
y = df["price"].to_numpy()
X = preprocessing.normalize(X)
y = preprocessing.normalize(y)
print(f"Feature Matrix Dimensions: {X.shape}\n Feature Matrix X: {X[0:5]}")
# === 3. Train/Test Split ===
