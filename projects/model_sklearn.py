# === 1. Import libraries ===
# - numpy
# - matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer

# === 2. Load & prepare the data ===
df = pd.read_csv("data/Housing.csv")
print(f"Columns: {df.columns}\n Rows: {len(df)}") 
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_Cols = df.select_dtypes(exclude=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Normalizer(), numerical_cols),
        ('cat', 'passthrough', categorical_Cols)
    ]
)


X = df[["area", "bathrooms"]].to_numpy()
y = df["price"].to_numpy()
print(f"Feature Matrix Dimensions: {X.shape}\n Feature Matrix X: {X[0:5]}")
# === 3. Train/Test Split ===
