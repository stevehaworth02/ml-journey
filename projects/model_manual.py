# === 1. Import libraries ===
# - numpy
# - matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

# === 2. Load & prepare the data ===
df = pd.read_csv("data/Housing.csv")
print(f"Columns: {df.columns}\n Rows: {len(df)}") 
X = df[["area", "bathrooms"]].to_numpy()
y = df["price"].to_numpy()
X = normalize(X)
print(f"Feature Matrix Dimensions: {X.shape}\n Feature Matrix X: {X[0:5]}")
# === 3. Train/Test Split ===
# - Use train_test_split from sklearn.model_selection (just for splitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
# === 4. Initialize parameters ===
# - Initialize weights (w) and bias (b) to 0 or small random values
# - Choose learning rate (alpha) and number of epochs
w = np.zeros((1, 2))
b = np.zeros((1,2))
print(w)
print(b)
lr = 1e3
epochs = 10

# === 5. Define functions ===
# - hypothesis(X, w, b): returns predicted y (X @ w + b)
# - compute_loss(y_pred, y_true): return MSE
# - compute_gradients(X, y, y_pred): return dw and db

# === 6. Training loop ===
# for each epoch:
#   - make predictions
#   - compute loss
#   - compute gradients
#   - update weights and bias
#   - (optional) store loss for plotting

# === 7. Evaluate ===
# - After training, compute final MSE on test set
# - Plot predictions vs actual prices
# - Plot loss over time if you tracked it

# === 8. Wrap up ===
# - Print learned weights and bias
# - Optionally compare with scikit-learn's LinearRegression
