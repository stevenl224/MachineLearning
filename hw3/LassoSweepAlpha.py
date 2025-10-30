import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix


# --- Feature Lifting Functions ---
def lift(x):
    """
    Lift a vector x ∈ R^d to include all second-order monomials.
    
    Returns a vector containing:
    1. All original features: x[0], x[1], ..., x[d-1]
    2. All products x[i] * x[j] for i >= j
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
    """Apply lift() to each row of dataset X."""
    return np.array([lift(x) for x in X])


# Number of samples
N = 1000

# Noise variance 
sigma = 0.01

# Feature dimension
d = 40

psfx = postfix(N, d, sigma) 

X = np.load("X" + psfx + ".npy")
y = np.load("y" + psfx + ".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape, "as well as %d labels." % y.shape[0])

# --- Lift the dataset to include second-order monomials ---
print("Lifting features to include second-order monomials...", end="")
X = liftDataset(X)
print(" done")
print("Lifted dataset now has n=%d samples, each with d=%d features" % X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples\n" % (X_train.shape[0], X_test.shape[0]))

# Generate range of alpha values to test
alpha_powers = np.linspace(-10, 10, 41)
alphas = 2.0 ** alpha_powers

print("Testing %d alpha values ranging from 2^%.1f to 2^%.1f\n" % (len(alphas), alpha_powers[0], alpha_powers[-1]))

# Setup cross-validation
cv = KFold(
    n_splits=5, 
    random_state=42,
    shuffle=True
)

# Store results
mean_cv_rmse = []
std_cv_rmse = []

# Perform cross-validation for each alpha
print("Running 5-fold cross-validation for each alpha...")
for i, alpha in enumerate(alphas):
    model = Lasso(alpha=alpha, max_iter=10000)
    
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    
    mean_rmse = -np.mean(scores)
    std_rmse = np.std(scores)
    
    mean_cv_rmse.append(mean_rmse)
    std_cv_rmse.append(std_rmse)
    
    if (i + 1) % 10 == 0:
        print("  Completed %d/%d alpha values..." % (i + 1, len(alphas)))

print("Cross-validation complete\n")

# Convert to arrays
mean_cv_rmse = np.array(mean_cv_rmse)
std_cv_rmse = np.array(std_cv_rmse)

# Find optimal alpha
best_idx = np.argmin(mean_cv_rmse)
best_alpha = alphas[best_idx]
best_alpha_power = alpha_powers[best_idx]
best_cv_rmse = mean_cv_rmse[best_idx]
best_cv_std = std_cv_rmse[best_idx]

print("Optimal alpha found: α = 2^%.2f = %f" % (best_alpha_power, best_alpha))
print("Cross-validation RMSE for optimal α: %f ± %f\n" % (best_cv_rmse, best_cv_std))

# Plot CV results with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(alpha_powers, mean_cv_rmse, yerr=std_cv_rmse, 
             fmt='o-', linewidth=2, markersize=5, capsize=3, 
             label='CV RMSE ± 1 std', color='blue', alpha=0.7)
plt.axvline(x=best_alpha_power, color='red', linestyle='--', 
            linewidth=2, label=f'Optimal α = 2^{best_alpha_power:.2f}')
plt.xlabel('log₂(α)', fontsize=12)
plt.ylabel('Cross-Validation RMSE', fontsize=12)
plt.title('Lasso Cross-Validation: RMSE vs Regularization Parameter', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('lasso_cv_alpha_sweep' + psfx + '.png', dpi=150)
print("Plot saved as 'lasso_cv_alpha_sweep%s.png'\n" % psfx)
plt.show()

# Train final model with optimal alpha on entire training set
print("Fitting Lasso model with optimal α over entire training set...", end="")
final_model = Lasso(alpha=best_alpha, max_iter=10000)
final_model.fit(X_train, y_train)
print(" done\n")

# Compute RMSE
rmse_train = rmse(y_train, final_model.predict(X_train))
rmse_test = rmse(y_test, final_model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f\n" % (rmse_train, rmse_test))

# Print model parameters
print("Model parameters:")
print("\t Intercept: %3.5f" % final_model.intercept_)

# Count and print non-zero coefficients
nonzero_coefs = np.abs(final_model.coef_) > 0.001
n_nonzero = np.sum(nonzero_coefs)

print("\nCoefficients with |β| > 0.001:")
for i, (coef, is_nonzero) in enumerate(zip(final_model.coef_, nonzero_coefs)):
    if is_nonzero:
        print("\t β%d: %3.5f" % (i, coef))

print("\nTotal non-zero coefficients: %d / %d" % (n_nonzero, len(final_model.coef_)))