import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eigvals, inv
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html



# Set random seed for reproducibility
np.random.seed(42)



def get_mean_and_covariance(X):
    """ Get the mean  mu in IR^d and covariance Sigma in IR^{d times d} from samples in matrix X in IR^{n x d}. Each row corresponds to a different sample"""
    e_mean = np.mean(X,0)
    e_cov = np.cov(X,rowvar = False)
    return (e_mean,e_cov)



def tau_range(X,g_pos,g_neg):
    """return the range of taus that one needs to span to create the ROC curve with these discriminant functions"""
    n,d = X.shape
    scores = [g_pos(X[i,:])-g_neg(X[i,:]) for i in range(n)]
    return min(scores),max(scores)
   

def predict(X,g_pos,g_neg,tau):
    """Generate predictions (+1/-1) over samples in X using discriminant functions g_pos, g_neg with threshold tau"""
    y_pred = []
    n,d = X.shape
    for i in range(n):
        x = X[i,:]
        y_pred.append(np.sign(g_pos(x)-g_neg(x)-tau))
    return y_pred     
    


def metrics(y_pred,y_true):
    """ Output the FPR and the TPR. These are defined in terms of the following quantities:
      
                 P = # entries y_true in that are  + 1
                 N = # entries y_true in that are  - 1
 
                 
                 TP = # entries for which  y_pred=+1 and y_true=+1  
                 FP = # entries for which  y_pred=-1 but y_true = +1  
                 TN = # entries for which y_pred=-1 and y_true = -1  
                 FN =  # entries for which y_pred=-1 but y_true = +1  


                 TPR = TP / P
                 FPR = FP / N
        Inputs are:
             - y_pred: predictions
             - y_true: ground truth values

        The return value is a tuple containing
             - #P,#N,#TP,#FP,#TN,#FN
    """

    pairs = [  (int(x),int(y))  for (x,y) in zip(y_pred,y_true)]
    new_pairs = [ (pred_label,pred_label*true_label)  for (pred_label,true_label) in pairs ]        
    

    TP = 1.*new_pairs.count( (1,1) )
    FP = 1.*new_pairs.count( (1,-1) )
    TN = 1.*new_pairs.count( (-1,1) )
    FN = 1.*new_pairs.count( (-1,-1) )
    P = TP+FN
    N = TN+FP

    TPR = 1.*TP/P
    FPR = 1.*FP/N
    return FPR,TPR

 
def naive_classifier(q,n):
    return -np.sign(np.random.rand(n)-q)


# NEW CLASSIFIER CLASSES
class IdentityCovaranceClassifier:
    """Gaussian classifier assuming identity covariance matrices for both classes"""
    
    def __init__(self):
        self.mu_pos = None
        self.mu_neg = None
        self.prior_pos = None
        self.prior_neg = None
        
    def fit(self, X_pos, X_neg):
        """Fit the classifier to training data"""
        self.mu_pos, _ = get_mean_and_covariance(X_pos)
        self.mu_neg, _ = get_mean_and_covariance(X_neg)
        
        n_pos, n_neg = len(X_pos), len(X_neg)
        n_total = n_pos + n_neg
        self.prior_pos = n_pos / n_total
        self.prior_neg = n_neg / n_total
        
    def discriminant_pos(self, x):
        """Discriminant function for positive class"""
        return -0.5 * np.sum((x - self.mu_pos)**2) + np.log(self.prior_pos)
    
    def discriminant_neg(self, x):
        """Discriminant function for negative class"""
        return -0.5 * np.sum((x - self.mu_neg)**2) + np.log(self.prior_neg)


class CommonCovarianceClassifier:
    """Gaussian classifier assuming common covariance matrix for both classes"""
    
    def __init__(self):
        self.mu_pos = None
        self.mu_neg = None
        self.sigma_common = None
        self.sigma_inv = None
        self.prior_pos = None
        self.prior_neg = None
        
    def fit(self, X_pos, X_neg):
        """Fit the classifier to training data"""
        self.mu_pos, cov_pos = get_mean_and_covariance(X_pos)
        self.mu_neg, cov_neg = get_mean_and_covariance(X_neg)
        
        n_pos, n_neg = len(X_pos), len(X_neg)
        n_total = n_pos + n_neg
        self.prior_pos = n_pos / n_total
        self.prior_neg = n_neg / n_total
        
        # Compute pooled covariance
        self.sigma_common = ((n_pos - 1) * cov_pos + (n_neg - 1) * cov_neg) / (n_total - 2)
        self.sigma_inv = inv(self.sigma_common)
        
    def discriminant_pos(self, x):
        """Discriminant function for positive class"""
        diff = x - self.mu_pos
        return -0.5 * np.dot(diff, np.dot(self.sigma_inv, diff)) + np.log(self.prior_pos)
    
    def discriminant_neg(self, x):
        """Discriminant function for negative class"""
        diff = x - self.mu_neg
        return -0.5 * np.dot(diff, np.dot(self.sigma_inv, diff)) + np.log(self.prior_neg)


def plot_decision_boundary(ax, classifier, xlims, ylims, label, color):
    """Plot decision boundary for a classifier"""
    x1_range = np.linspace(xlims[0], xlims[1], 100)
    x2_range = np.linspace(ylims[0], ylims[1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i,j], X2[i,j]])
            Z[i,j] = classifier.discriminant_pos(x) - classifier.discriminant_neg(x)
    
    ax.contour(X1, X2, Z, levels=[0], colors=color, linewidths=2, linestyles='--')
    
    # Add a dummy line for legend
    ax.plot([], [], color=color, linestyle='--', linewidth=2, label=f'{label} Boundary')


def calculate_accuracy(y_pred, y_true):
    """Calculate classification accuracy"""
    correct = sum([1 for pred, true in zip(y_pred, y_true) if pred == true])
    return correct / len(y_true)


p = 0.5 # prior probability of positive class
n = 1000 # total number of samples


if __name__=="__main__":
    # Generate two multivariate Gaussian random variable objects
    mean_pos, cov_pos = [1, 3], [[1, 0.8], [0.8, 1]]
    e=eigvals(cov_pos)
    print("Eigenvalues of cov_pos are:",e)
    rv_pos = multivariate_normal(mean_pos, cov_pos)

    mean_neg, cov_neg = [4, 4], [[0.99, 0.81], [0.81, 1.01]]
    e=eigvals(cov_neg)
    print("Eigenvalues of cov_neg are:",e)
    rv_neg = multivariate_normal(mean_neg, cov_neg)


    prior_pos = p
    prior_neg = 1-p

    # Generate samples
    n_pos = np.random.binomial(n,p)
    n_neg = n - n_pos
    X_pos = rv_pos.rvs(size = n_pos)
    X_neg = rv_neg.rvs(size = n_neg)


    # Create a scatter plot for the samples
    fig	 = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_pos[:, 0], X_pos[:, 1], label='Sea Bass (+)', alpha=0.7,c="blue")
    ax.scatter(X_neg[:, 0], X_neg[:, 1], label='Salmon (-)', alpha=0.7, c="orange")

    # Add contours of the pdfs
    x1, x2 = np.mgrid[-1:8:.01, -1:6:.01]
    positions = np.dstack((x1, x2))
    ax.contour(x1, x2, rv_pos.pdf(positions),levels=5,colors="blue")
    ax.contour(x1, x2, rv_neg.pdf(positions), levels=5, colors="orange")


    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()


    plt.savefig("two-gaussians.pdf", format="pdf", bbox_inches="tight")

   
    # Estimate the mean, covariance, and prior for each Gaussian from samples
    e_mean_pos, e_cov_pos = get_mean_and_covariance(X_pos)
    e_mean_neg, e_cov_neg = get_mean_and_covariance(X_neg)
    e_prior_pos = 1. * n_pos / n
    e_prior_neg = 1. * n_neg / n

    print ("Positive class estimates:", e_mean_pos,e_cov_pos,e_prior_pos)
    print ("Negative class estimates:", e_mean_neg,e_cov_neg,e_prior_neg)


    # Join samples to create a classification dataset
    X = np.concatenate((X_pos,X_neg))
    y = np.concatenate( (np.ones(n_pos),-np.ones(n_neg)) )
 
    
    #create ROC Curve for naive classifier
    FPR_naive_list=[]
    TPR_naive_list=[]
    delta = (1.0-0.0)/10
    print('Scanning tau for naive...')
    for q in np.arange(0.,1+delta,delta):
        #generate predictions
        y_pred = naive_classifier(q,n)

        #compute FPR and TPR
        FPR,TPR = metrics(y_pred,y)
        print('FPR =',FPR,'TPR =',TPR)
        FPR_naive_list.append(FPR)
        TPR_naive_list.append(TPR)
    print('...done with',len(FPR_naive_list),'measurements')
    
    # =========================
    # NEW CLASSIFIERS SECTION
    # =========================
    
    print("\n" + "="*50)
    print("FITTING NEW CLASSIFIERS")
    print("="*50)
    
    # Fit Identity Covariance Classifier
    identity_clf = IdentityCovaranceClassifier()
    identity_clf.fit(X_pos, X_neg)
    print(f"Identity Classifier - Estimated means:")
    print(f"  Positive: {identity_clf.mu_pos}")
    print(f"  Negative: {identity_clf.mu_neg}")
    print(f"  Priors: {identity_clf.prior_pos:.3f}, {identity_clf.prior_neg:.3f}")
    
    # Fit Common Covariance Classifier
    common_clf = CommonCovarianceClassifier()
    common_clf.fit(X_pos, X_neg)
    print(f"\nCommon Covariance Classifier - Estimated parameters:")
    print(f"  Positive mean: {common_clf.mu_pos}")
    print(f"  Negative mean: {common_clf.mu_neg}")
    print(f"  Common covariance:\n{common_clf.sigma_common}")
    print(f"  Priors: {common_clf.prior_pos:.3f}, {common_clf.prior_neg:.3f}")
    
    # Generate ROC curves for new classifiers
    print("\n" + "="*50)
    print("GENERATING ROC CURVES FOR NEW CLASSIFIERS")
    print("="*50)
    
    # Identity Covariance ROC
    tau_min, tau_max = tau_range(X, identity_clf.discriminant_pos, identity_clf.discriminant_neg)
    FPR_identity_list = []
    TPR_identity_list = []
    delta_tau = (tau_max - tau_min) / 50
    
    print(f'Identity Classifier - Tau range: [{tau_min:.3f}, {tau_max:.3f}]')
    for tau in np.arange(tau_min, tau_max + delta_tau, delta_tau):
        y_pred = predict(X, identity_clf.discriminant_pos, identity_clf.discriminant_neg, tau)
        FPR, TPR = metrics(y_pred, y)
        FPR_identity_list.append(FPR)
        TPR_identity_list.append(TPR)
    print(f'...done with {len(FPR_identity_list)} measurements')
    
    # Common Covariance ROC
    tau_min, tau_max = tau_range(X, common_clf.discriminant_pos, common_clf.discriminant_neg)
    FPR_common_list = []
    TPR_common_list = []
    delta_tau = (tau_max - tau_min) / 50
    
    print(f'Common Covariance Classifier - Tau range: [{tau_min:.3f}, {tau_max:.3f}]')
    for tau in np.arange(tau_min, tau_max + delta_tau, delta_tau):
        y_pred = predict(X, common_clf.discriminant_pos, common_clf.discriminant_neg, tau)
        FPR, TPR = metrics(y_pred, y)
        FPR_common_list.append(FPR)
        TPR_common_list.append(TPR)
    print(f'...done with {len(FPR_common_list)} measurements')
    
    # Calculate optimal accuracies (at tau=0)
    print("\n" + "="*50)
    print("CLASSIFICATION ACCURACIES (at tau=0)")
    print("="*50)
    
    # Identity classifier accuracy
    y_pred_identity = predict(X, identity_clf.discriminant_pos, identity_clf.discriminant_neg, 0)
    acc_identity = calculate_accuracy(y_pred_identity, y)
    print(f"Identity Covariance Classifier Accuracy: {acc_identity:.3f}")
    
    # Common covariance classifier accuracy
    y_pred_common = predict(X, common_clf.discriminant_pos, common_clf.discriminant_neg, 0)
    acc_common = calculate_accuracy(y_pred_common, y)
    print(f"Common Covariance Classifier Accuracy: {acc_common:.3f}")
    
    # Enhanced ROC Plot with all three classifiers
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(FPR_naive_list, TPR_naive_list, linestyle='dashed', color='yellow', 
            linewidth=2, label='Naive', marker='o', markersize=4)
    ax.plot(FPR_identity_list, TPR_identity_list, linestyle='-', color='red', 
            linewidth=2, label='Identity Covariance', marker='s', markersize=3)
    ax.plot(FPR_common_list, TPR_common_list, linestyle='-', color='green', 
            linewidth=2, label='Common Covariance', marker='^', markersize=3)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

    ax.set_xlabel('FPR (False Positive Rate)', fontsize=12)
    ax.set_ylabel('TPR (True Positive Rate)', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.savefig("ROC.pdf", format="pdf", bbox_inches="tight")
    
    # Decision Boundary Visualization
    print("\n" + "="*50)
    print("CREATING DECISION BOUNDARY VISUALIZATION")
    print("="*50)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot data points
    ax.scatter(X_pos[:, 0], X_pos[:, 1], label='Sea Bass (+)', alpha=0.6, c="blue", s=30)
    ax.scatter(X_neg[:, 0], X_neg[:, 1], label='Salmon (-)', alpha=0.6, c="orange", s=30)
    
    # Add PDF contours (lighter)
    ax.contour(x1, x2, rv_pos.pdf(positions), levels=3, colors="blue", alpha=0.3, linewidths=1)
    ax.contour(x1, x2, rv_neg.pdf(positions), levels=3, colors="orange", alpha=0.3, linewidths=1)
    
    # Plot decision boundaries
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    plot_decision_boundary(ax, identity_clf, xlims, ylims, 'Identity', 'red')
    plot_decision_boundary(ax, common_clf, xlims, ylims, 'Common', 'green')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Decision Boundaries and Data Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.savefig("decision-boundaries.pdf", format="pdf", bbox_inches="tight")
    
    print("\nAnalysis complete! Generated files:")
    print("- two-gaussians.pdf: Original data visualization")
    print("- ROC.pdf: ROC curves for all three classifiers")
    print("- decision-boundaries.pdf: Decision boundary visualization")
    print("\nBoth new classifiers have linear decision boundaries as expected!")