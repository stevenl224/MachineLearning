"""
Categorical Naïve Bayes Classifier for Mushroom Dataset - Extreme Small Sample Experiment
==========================================================================================
This script implements a complete pipeline for binary classification of mushrooms
(edible vs poisonous) using only 1% of data for training (99% for testing) to evaluate
model performance under severe data scarcity.
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
# STEP 2: TRAIN-TEST SPLIT (1% TRAINING, 99% TESTING)
# ============================================================================

def create_train_test_split(X, y, test_size=0.99, random_state=42):
    """
    Create 1-99 train-test split for extreme small sample experiment.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set (default: 0.99)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("Creating Train-Test Split (EXTREME SMALL SAMPLE: 1% Train, 99% Test)")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train ratio: {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2%}")
    print(f"Test ratio: {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2%}")
    
    print("\n⚠️  WARNING: Using only 1% of data for training!")
    print("   This experiment tests model performance under extreme data scarcity.")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 3: DETECT UNSEEN CATEGORIES
# ============================================================================

def detect_unseen_categories(X_train, X_test, feature_names):
    """
    Detect which features have unseen categories in test set.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        feature_names: List of feature names
    
    Returns:
        unseen_info: Dictionary with information about unseen categories
    """
    print("\n" + "=" * 70)
    print("Analyzing Unseen Categories in Test Set")
    print("=" * 70)
    
    unseen_info = {
        'features_with_unseen': [],
        'unseen_counts': [],
        'total_unseen': 0
    }
    
    n_features = X_train.shape[1]
    
    for feat_idx in range(n_features):
        train_categories = set(X_train[:, feat_idx])
        test_categories = set(X_test[:, feat_idx])
        unseen_categories = test_categories - train_categories
        
        if len(unseen_categories) > 0:
            unseen_info['features_with_unseen'].append(feature_names[feat_idx])
            unseen_info['unseen_counts'].append(len(unseen_categories))
            unseen_info['total_unseen'] += len(unseen_categories)
            
            print(f"\n⚠️  Feature '{feature_names[feat_idx]}':")
            print(f"   - Training categories: {sorted(train_categories)}")
            print(f"   - Test categories: {sorted(test_categories)}")
            print(f"   - Unseen in test: {sorted(unseen_categories)}")
            print(f"   - Count: {len(unseen_categories)} unseen category(ies)")
    
    if unseen_info['total_unseen'] == 0:
        print("\n✓ No unseen categories detected! All test categories appear in training.")
    else:
        print(f"\n⚠️  SUMMARY: {len(unseen_info['features_with_unseen'])} features have unseen categories")
        print(f"   Total unseen categories across all features: {unseen_info['total_unseen']}")
    
    return unseen_info


# ============================================================================
# STEP 4: SAFE PREDICTION WITH CLIPPING
# ============================================================================

def clip_unseen_categories(X_train, X_test):
    """
    Clip test set categories to maximum value seen in training set.
    This prevents index out of bounds errors in CategoricalNB.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
    
    Returns:
        X_test_clipped: Test set with clipped category values
        clipping_info: Information about what was clipped
    """
    X_test_clipped = X_test.copy()
    clipping_info = {'features_clipped': [], 'samples_affected': 0}
    
    n_features = X_train.shape[1]
    samples_affected = set()
    
    for feat_idx in range(n_features):
        max_train_category = X_train[:, feat_idx].max()
        
        # Find samples with categories exceeding training maximum
        exceeds_mask = X_test[:, feat_idx] > max_train_category
        
        if exceeds_mask.any():
            clipping_info['features_clipped'].append(feat_idx)
            samples_affected.update(np.where(exceeds_mask)[0])
            
            # Clip to maximum training category
            X_test_clipped[exceeds_mask, feat_idx] = max_train_category
    
    clipping_info['samples_affected'] = len(samples_affected)
    
    return X_test_clipped, clipping_info


# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH ERROR HANDLING
# ============================================================================

def tune_categorical_nb(X_train, X_test, y_train, y_test, feature_names):
    """
    Train CategoricalNB with different alpha values and evaluate performance.
    Includes robust error handling for small sample issues.
    
    Args:
        X_train, X_test, y_train, y_test: Train-test split data
        feature_names: List of feature names
    
    Returns:
        results: Dictionary containing alpha values and corresponding metrics
        best_model: The model with the best ROC AUC score
        best_alpha: The alpha value that maximizes ROC AUC
    """
    print("\n" + "=" * 70)
    print("Hyperparameter Tuning: Alpha (Smoothing Parameter)")
    print("=" * 70)
    
    # Detect unseen categories
    unseen_info = detect_unseen_categories(X_train, X_test, feature_names)
    
    # Clip test set to handle unseen categories
    print("\n" + "=" * 70)
    print("Applying Safety Measures: Clipping Unseen Categories")
    print("=" * 70)
    X_test_safe, clipping_info = clip_unseen_categories(X_train, X_test)
    
    if len(clipping_info['features_clipped']) > 0:
        print(f"✓ Clipped {len(clipping_info['features_clipped'])} feature(s)")
        print(f"✓ Affected {clipping_info['samples_affected']} test sample(s)")
        print("  (Unseen categories mapped to maximum training category)")
    else:
        print("✓ No clipping needed - all categories within training range")
    
    # Generate alpha values from 2^-15 to 2^5
    alpha_values = np.logspace(-15, 5, num=40, base=2)
    print(f"\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)
    print(f"Testing {len(alpha_values)} alpha values")
    print(f"Alpha range: [{alpha_values.min():.2e}, {alpha_values.max():.2e}]")
    
    # Storage for results
    results = {
        'alpha': [],
        'roc_auc': [],
        'accuracy': [],
        'f1': []
    }
    
    best_roc_auc = 0
    best_model = None
    best_alpha = None
    successful_models = 0
    failed_models = 0
    
    print("\nTraining models with error handling...")
    for i, alpha in enumerate(alpha_values):
        try:
            # Train Categorical Naïve Bayes classifier
            clf = CategoricalNB(alpha=alpha)
            clf.fit(X_train, y_train)
            
            # Make predictions with error handling
            try:
                y_pred = clf.predict(X_test_safe)
                y_pred_proba = clf.predict_proba(X_test_safe)[:, 1]
                
                # Compute metrics
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Store results
                results['alpha'].append(alpha)
                results['roc_auc'].append(roc_auc)
                results['accuracy'].append(accuracy)
                results['f1'].append(f1)
                
                # Track best model
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model = clf
                    best_alpha = alpha
                
                successful_models += 1
                
            except Exception as pred_error:
                print(f"  ⚠️  Prediction failed for alpha={alpha:.2e}: {str(pred_error)}")
                failed_models += 1
                continue
                
        except Exception as fit_error:
            print(f"  ⚠️  Training failed for alpha={alpha:.2e}: {str(fit_error)}")
            failed_models += 1
            continue
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(alpha_values)} alpha values tested "
                  f"({successful_models} successful, {failed_models} failed)")
    
    print(f"\n✓ Training complete!")
    print(f"  Successful models: {successful_models}/{len(alpha_values)}")
    print(f"  Failed models: {failed_models}/{len(alpha_values)}")
    
    if successful_models == 0:
        print("\n" + "=" * 70)
        print("❌ EXPERIMENT FAILED")
        print("=" * 70)
        print("All alpha values failed to produce valid models.")
        print("This likely indicates that 1% training data is insufficient for this")
        print("dataset configuration. Possible reasons:")
        print("  • Too few samples per class")
        print("  • Too many unseen category combinations")
        print("  • Extreme class imbalance in tiny training set")
        return None, None, None
    
    if best_model is not None:
        print(f"\n✓ Best ROC AUC: {best_roc_auc:.6f} at alpha={best_alpha:.2e}")
    
    return results, best_model, best_alpha


# ============================================================================
# STEP 6: VISUALIZATION
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
    
    if results is None or len(results['alpha']) == 0:
        print("⚠️  No results to plot (all models failed)")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Categorical Naïve Bayes Performance vs Smoothing Parameter (α)\n' + 
                 '⚠️ EXTREME EXPERIMENT: 1% Training Data, 99% Test Data',
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
        
        # Set reasonable y-limits
        y_min = max(0, min(results[metric]) - 0.05)
        y_max = min(1, max(results[metric]) + 0.05)
        ax.set_ylim([y_min, y_max])
        
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
    plt.savefig('mushroom_nb_metrics_1pct.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'mushroom_nb_metrics_1pct.png'")
    plt.show()


# ============================================================================
# STEP 7: PRINT MODEL PARAMETERS
# ============================================================================

def print_model_parameters(model, alpha, feature_names):
    """
    Print the parameters of the best model.
    
    Args:
        model: Trained CategoricalNB model
        alpha: The alpha value used
        feature_names: List of feature names
    """
    if model is None:
        print("\n⚠️  No model available to display parameters (all training failed)")
        return
    
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
    print("█" + " " * 5 + "MUSHROOM CLASSIFICATION - EXTREME SMALL SAMPLE EXPERIMENT" + " " * 6 + "█")
    print("█" + " " * 15 + "(1% Training, 99% Testing)" + " " * 27 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Step 1: Load and preprocess data
    X, y, feature_names, encoders = load_and_preprocess_mushroom_data()
    
    # Step 2: Create train-test split (1% train, 99% test)
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=0.99, random_state=42
    )
    
    # Step 3: Hyperparameter tuning with error handling
    results, best_model, best_alpha = tune_categorical_nb(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Check if experiment succeeded
    if results is None or best_model is None:
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + " " * 20 + "EXPERIMENT INCOMPLETE" + " " * 27 + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70 + "\n")
        print("The experiment could not complete successfully with 1% training data.")
        print("Consider increasing the training set size for meaningful results.")
        return
    
    # Step 4: Visualize results
    plot_metrics(results)
    
    # Step 5: Print optimal model parameters
    print_model_parameters(best_model, best_alpha, feature_names)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings (1% Training Data Experiment):")
    print(f"  • Best Alpha: {best_alpha:.2e}")
    print(f"  • Best ROC AUC: {max(results['roc_auc']):.6f}")
    print(f"  • Best Accuracy: {max(results['accuracy']):.6f}")
    print(f"  • Best F1 Score: {max(results['f1']):.6f}")
    print(f"  • Training samples: {X_train.shape[0]}")
    print(f"  • Test samples: {X_test.shape[0]}")
    
    print("\n📊 Interpretation:")
    if max(results['roc_auc']) > 0.85:
        print("   ✓ Surprisingly good performance despite minimal training data!")
        print("   The strong results suggest the categorical features are highly")
        print("   informative and the mushroom classification task has clear patterns.")
    elif max(results['roc_auc']) > 0.7:
        print("   ~ Moderate performance with only 1% training data.")
        print("   The model learned some patterns but would benefit from more examples.")
    else:
        print("   ⚠️  Poor performance, as expected with such limited training data.")
        print("   The model struggles to generalize from so few examples.")
    
    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    main()