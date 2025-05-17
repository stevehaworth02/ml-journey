# === 1. Import libraries ===
# - numpy
# - matplotlib.pyplot

# === 2. Load & prepare the data ===
# - Read Housing.csv
# - Select relevant columns (e.g., 'area', 'price')
# - Normalize features (optional but recommended)
# - Split into X (features), y (target)

# === 3. Train/Test Split ===
# - Use train_test_split from sklearn.model_selection (just for splitting)

# === 4. Initialize parameters ===
# - Initialize weights (w) and bias (b) to 0 or small random values
# - Choose learning rate (alpha) and number of epochs

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
