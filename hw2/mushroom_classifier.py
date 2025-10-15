"""
Categorical Naïve Bayes Classifier for Mushroom Dataset
========================================================
This script implements a complete pipeline for binary classification of mushrooms
(edible vs poisonous) using sklearn's CategoricalNB classifier with hyperparameter tuning.
"""

from ucimlrepo import fetch_ucirepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PREPROCESS THE MUSHROOM DATASET
# ============================================================================

def load_and_preprocess_mushroom_data():
    """
    Load the mushroom dataset and preprocess it for CategoricalNB.
    
    Returns:
        X: Feature matrix (encoded categorical features)
        y: Target vector (0=edible, 1=poisonous)
        feature_names: List of feature names
        encoders: Dictionary of LabelEncoders for each feature
    """
    print("=" * 70)
    print("Loading Mushroom Dataset")
    print("=" * 70)
    
    # Column names for the mushroom dataset
    column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    # Load the dataset from the UCI repository
mushroom = fetch_ucirepo(id=73)
df = pd.concat([mushroom.data.features, mushroom.data.targets], axis=1)
df.columns = [*mushroom.data.features.columns, 'class']


    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    
    # Handle missing values (represented as '?')
    missing_count = (df == '?').sum().sum()
    print(f"Missing values ('?'): {missing_count}")
    
    # Replace '?' with the most frequent value in each column
    for col in df.columns:
        if '?' in df[col].values:
            mode_value = df[col][df[col] != '?'].mode()[0]
            df[col] = df[col].replace('?', mode_value)
            print(f"  - Replaced '?' in {col} with mode: {mode_value}")
    
    # Separate features and target
    y = df['class']
    X = df.drop('class', axis=1)
    feature_names = X.columns.tolist()
    
    # Encode target variable: 'e'=edible (0), 'p'=poisonous (1)
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    print(f"\nTarget distribution:")
    print(f"  - Edible (0): {(y_encoded == 0).sum()} samples")
    print(f"  - Poisonous (1): {(y_encoded == 1).sum()} samples")
    
    # Encode categorical features
    # CategoricalNB requires features to be integers starting from 0
    X_encoded = pd.DataFrame()
    encoders = {}
    
    print(f"\nEncoding {len(feature_names)} categorical features...")
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} categories")
    
    return X_encoded.values, y_encoded, feature_names, encoders


# ============================================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================================

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create 80-20 train-test split with reproducibility.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("Creating Train-Test Split")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train ratio: {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2%}")
    print(f"Test ratio: {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2%}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 3: HYPERPARAMETER TUNING
# ============================================================================

def tune_categorical_nb(X_train, X_test, y_train, y_test):
    """
    Train CategoricalNB with different alpha values and evaluate performance.
    
    Args:
        X_train, X_test, y_train, y_test: Train-test split data
    
    Returns:
        results: Dictionary containing alpha values and corresponding metrics
        best_model: The model with the best ROC AUC score
        best_alpha: The alpha value that maximizes ROC AUC
    """
    print("\n" + "=" * 70)
    print("Hyperparameter Tuning: Alpha (Smoothing Parameter)")
    print("=" * 70)
    
    # Generate alpha values from 2^-15 to 2^5
    alpha_values = np.logspace(-15, 5, num=40, base=2)
    print(f"Testing {len(alpha_values)} alpha values")
    print(f"Alpha range: [{alpha_values.min():.2e}, {alpha_values.max():.2e}]")
    
    # Storage for results
    results = {
        'alpha': alpha_values,
        'roc_auc': [],
        'accuracy': [],
        'f1': []
    }
    
    best_roc_auc = 0
    best_model = None
    best_alpha = None
    
    print("\nTraining models...")
    for i, alpha in enumerate(alpha_values):
        # Train Categorical Naïve Bayes classifier
        clf = CategoricalNB(alpha=alpha)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results['roc_auc'].append(roc_auc)
        results['accuracy'].append(accuracy)
        results['f1'].append(f1)
        
        # Track best model
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = clf
            best_alpha = alpha
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(alpha_values)} models trained")
    
    print(f"\nBest ROC AUC: {best_roc_auc:.6f} at alpha={best_alpha:.2e}")
    
    return results, best_model, best_alpha


# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def plot_metrics(results):
    """
    Create publication-quality plots of metrics vs alpha.
    
    Args:
        results: Dictionary containing alpha values and metrics
    """
    print("\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Categorical Naïve Bayes Performance vs Smoothing Parameter (α)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['roc_auc', 'accuracy', 'f1']
    titles = ['ROC AUC Score', 'Accuracy', 'F1 Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        ax.plot(results['alpha'], results[metric], 
                linewidth=2.5, color=color, marker='o', markersize=4)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Alpha (Smoothing Parameter)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([min(results[metric]) - 0.01, max(results[metric]) + 0.01])
        
        # Highlight best value
        best_idx = np.argmax(results[metric])
        ax.axvline(results['alpha'][best_idx], color=color, 
                  linestyle='--', alpha=0.5, linewidth=1.5)
        ax.plot(results['alpha'][best_idx], results[metric][best_idx],
               'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Add annotation for best value
        ax.annotate(f'Best: {results[metric][best_idx]:.4f}',
                   xy=(results['alpha'][best_idx], results[metric][best_idx]),
                   xytext=(20, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('mushroom_nb_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'mushroom_nb_metrics.png'")
    plt.show()


# ============================================================================
# STEP 5: PRINT MODEL PARAMETERS
# ============================================================================

def print_model_parameters(model, alpha, feature_names):
    """
    Print the parameters of the best model.
    
    Args:
        model: Trained CategoricalNB model
        alpha: The alpha value used
        feature_names: List of feature names
    """
    print("\n" + "=" * 70)
    print("OPTIMAL MODEL PARAMETERS")
    print("=" * 70)
    
    print(f"\nBest Alpha (Smoothing Parameter): {alpha:.10e}")
    print(f"  → This corresponds to 2^{np.log2(alpha):.2f}")
    
    print("\n" + "-" * 70)
    print("CLASS PRIORS")
    print("-" * 70)
    print(f"Class 0 (Edible):    P(y=0) = {model.class_prior_[0]:.6f}")
    print(f"Class 1 (Poisonous): P(y=1) = {model.class_prior_[1]:.6f}")
    
    print("\n" + "-" * 70)
    print("FEATURE LOG PROBABILITIES")
    print("-" * 70)
    print("\nShape of feature_log_prob_: ", model.feature_log_prob_.shape)
    print("  → (n_classes, n_features, n_categories)")
    print(f"  → ({model.feature_log_prob_.shape[0]} classes, "
          f"{model.feature_log_prob_.shape[1]} features)")
    
    print("\nFeature probability dimensions:")
    for i, feature_name in enumerate(feature_names):
        n_categories = model.category_count_[0][i].shape[0]
        print(f"  {i+1:2d}. {feature_name:30s} → {n_categories:2d} categories")
    
    print("\n" + "-" * 70)
    print("SAMPLE FEATURE PROBABILITIES (First 3 features)")
    print("-" * 70)
    
    for feat_idx in range(min(3, len(feature_names))):
        print(f"\nFeature: {feature_names[feat_idx]}")
        n_categories = model.feature_log_prob_[0][feat_idx].shape[0]
        
        for class_idx in range(2):
            class_label = "Edible" if class_idx == 0 else "Poisonous"
            print(f"  Class {class_idx} ({class_label}):")
            log_probs = model.feature_log_prob_[class_idx][feat_idx]
            probs = np.exp(log_probs)
            
            for cat_idx in range(min(5, n_categories)):
                print(f"    Category {cat_idx}: P(x={cat_idx}|y={class_idx}) = "
                      f"{probs[cat_idx]:.6f} (log: {log_probs[cat_idx]:.4f})")
            
            if n_categories > 5:
                print(f"    ... ({n_categories - 5} more categories)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that runs the complete pipeline.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 10 + "MUSHROOM CLASSIFICATION WITH CATEGORICAL NB" + " " * 15 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Step 1: Load and preprocess data
    X, y, feature_names, encoders = load_and_preprocess_mushroom_data()
    
    # Step 2: Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 3: Hyperparameter tuning
    results, best_model, best_alpha = tune_categorical_nb(
        X_train, X_test, y_train, y_test
    )
    
    # Step 4: Visualize results
    plot_metrics(results)
    
    # Step 5: Print optimal model parameters
    print_model_parameters(best_model, best_alpha, feature_names)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • Best Alpha: {best_alpha:.2e}")
    print(f"  • Best ROC AUC: {max(results['roc_auc']):.6f}")
    print(f"  • Best Accuracy: {max(results['accuracy']):.6f}")
    print(f"  • Best F1 Score: {max(results['f1']):.6f}")
    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    main()