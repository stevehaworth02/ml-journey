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
w = np.zeros(2)
b = 0.0
print(w)
print(b)
lr = 1e3
epochs = 10
# === 5. Define functions ===
# - hypothesis(X, w, b): returns predicted y (X @ w + b)
# - compute_loss(y_pred, y_true): return MSE
# - compute_gradients(X, y, y_pred): return dw and db
def hypothesis(input_matrix, weight_vector, bias_vector):
    y_pred = input_matrix @ weight_vector.T + bias_vector
    return y_pred
def compute_loss(predicted, actual):
    loss = np.mean(predicted - actual)**2
    return loss
def compute_gradient(X, y, y_pred):
    error = y_pred - y
    dw = ((1/len(X)) * (error.T @ X))
    bw = np.mean(error)
    return dw, bw
# === 6. Training loop ===
# for each epoch:
#   - make predictions
#   - compute loss
#   - compute gradients
#   - update weights and bias
#   - (optional) store loss for plotting
losses = []
for epoch in range(epochs):
    predicted_y = hypothesis(X_train, w, b)
    loss = compute_loss(predicted_y, y_train)
    dw, bw = compute_gradient(X_train, y_train, predicted_y)
    w -= lr * dw
    b -= lr * bw
    losses.append(loss)
    print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# === 7. Evaluate ===
# - After training, compute final MSE on test set
# - Plot predictions vs actual prices
# - Plot loss over time if you tracked it

# === 8. Wrap up ===
# - Print learned weights and bias
# - Optionally compare with scikit-learn's LinearRegression
