# === 1. Import libraries ===
# - numpy
# - matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

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
y = normalize(y)
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
lr = 1e-3
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
y_pred_test = hypothesis(X_test, w, b)
test_loss = compute_loss(y_pred_test, y_test)
print(test_loss)


# === 8. Wrap up ===


# === Plot 3D regression surface + ACTUAL TEST DATA GRID ===

# Step 1: Use min/max from your actual X_test
area_min, area_max = X_test[:, 0].min(), X_test[:, 0].max()
bath_min, bath_max = X_test[:, 1].min(), X_test[:, 1].max()

area_vals = np.linspace(area_min, area_max, 50)
bathroom_vals = np.linspace(bath_min, bath_max, 50)
A, B = np.meshgrid(area_vals, bathroom_vals)

# Step 2: Create input pairs from grid and predict using hypothesis()
grid_inputs = np.column_stack((A.ravel(), B.ravel()))
Z = hypothesis(grid_inputs, w, b).reshape(A.shape)

# Step 3: Plot surface + your actual test data points
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot regression surface
ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.7)

# Overlay test points (actual data)
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='r', label='Actual Test Points', s=20)

# Labels + legend
ax.set_xlabel("Normalized Area")
ax.set_ylabel("Normalized Bathrooms")
ax.set_zlabel("Normalized Price")
ax.set_title("Regression Surface with Real Test Data Overlay")
ax.legend()

plt.tight_layout()
plt.show()
