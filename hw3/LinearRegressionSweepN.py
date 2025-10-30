import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
import os, runpy
from pathlib import Path

# --- Feature Lifting Functions ---
def lift(x):
    """
    Lift a vector x ∈ R^d to include all second-order monomials.
    
    Returns a vector containing:
    1. All original features: x[0], x[1], ..., x[d-1]
    2. All products x[i] * x[j] for i >= j, in order:
       x[0]*x[0], x[1]*x[0], x[1]*x[1], x[2]*x[0], x[2]*x[1], x[2]*x[2], ..., x[d-1]*x[d-1]
    
    Args:
        x: 1D numpy array of shape (d,)
    
    Returns:
        lifted_x: 1D numpy array containing original features and all second-order terms
    """
    d = len(x)
    
    # Start with original features
    lifted_features = list(x)
    
    # Add all second-order monomials x[i] * x[j] for i >= j
    for i in range(d):
        for j in range(i + 1):
            lifted_features.append(x[i] * x[j])
    
    return np.array(lifted_features)


def liftDataset(X):
    """
    Apply lift() to each row of dataset X.
    
    Args:
        X: 2D numpy array of shape (n, d) where n is number of samples
    
    Returns:
        X_lifted: 2D numpy array of shape (n, d_lifted) where d_lifted = d + d(d+1)/2
    """
    return np.array([lift(x) for x in X])


# --- Generate & Load Dataset ---
BASE_DIR = Path(__file__).resolve().parent 
runpy.run_path(str(BASE_DIR / "data_generator.py"), run_name="__main__")

# Find the most recent X*.npy and y*.npy in this folder
x_candidates = sorted(BASE_DIR.glob("X*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
y_candidates = sorted(BASE_DIR.glob("y*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)

if not x_candidates or not y_candidates:
    raise FileNotFoundError(f"No generated dataset found in {BASE_DIR}. Did data_generator.py save X*.npy and y*.npy?")

x_path, y_path = x_candidates[0], y_candidates[0]
X = np.load(x_path)
y = np.load(y_path)

# Derive postfix (e.g., 'X_N_1000_d_5_sig_0_01.npy' -> '_N_1000_d_5_sig_0_01')
psfx = os.path.splitext(x_path.name)[0][1:]

print(f"Loaded dataset '{x_path.name}' and '{y_path.name}' from {BASE_DIR}")
print("Dataset has n=%d samples, each with d=%d features," % X.shape, "as well as %d labels." % y.shape[0])

# --- Lift the dataset to include second-order monomials ---
print("Lifting features to include second-order monomials...", end="")
X = liftDataset(X)
print(" done")
print("Lifted dataset now has n=%d samples, each with d=%d features\n" % X.shape)

# --- Split dataset 70% train / 30% test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples\n" % (X_train.shape[0], X_test.shape[0]))

# Define fractions of training data to use
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Store results
n_samples_list = []
train_rmse_list = []
test_rmse_list = []

# --- Train models with different fractions of training data ---
for fr in fractions:
    # Calculate number of samples to use
    n_samples = int(fr * X_train.shape[0])
    n_samples_list.append(n_samples)
    
    # Use first n_samples of training data
    X_train_subset = X_train[:n_samples]
    y_train_subset = y_train[:n_samples]
    
    # Train model
    model = LinearRegression()
    print("Training with %d samples (%.0f%% of training data)..." % (n_samples, fr * 100), end="")
    model.fit(X_train_subset, y_train_subset)
    print(" done")
    
    # Compute RMSE on training subset
    rmse_train = rmse(y_train_subset, model.predict(X_train_subset))
    train_rmse_list.append(rmse_train)
    
    # Compute RMSE on full test set
    rmse_test = rmse(y_test, model.predict(X_test))
    test_rmse_list.append(rmse_test)
    
    print("  Train RMSE = %f, Test RMSE = %f" % (rmse_train, rmse_test))
    
    # Print model parameters
    print("  Model parameters: Intercept: %3.5f" % model.intercept_, end="")
    for i, val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i, val), end="")
    print("\n")

# --- Plot learning curves ---
plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, train_rmse_list, 'o-', label='Training RMSE', linewidth=2, markersize=8)
plt.plot(n_samples_list, test_rmse_list, 's-', label='Test RMSE', linewidth=2, markersize=8)
plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Learning Curves: RMSE vs Training Set Size (Lifted Features)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('learning_curves_lifted' + psfx + '.png', dpi=150)
print("Plot saved as 'learning_curves_lifted%s.png'" % psfx)
plt.show()