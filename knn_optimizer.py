#Cross Validation for KNN
from mnist_loader import MNISTLoader
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
from collections import defaultdict

class KNNOptimizer:
    def __init__(self, X_train, y_train, k_range=range(1, 16, 2)):
        """
        Initialize KNN optimizer.
        
        Args:
            X_train: Training features
            y_train: Training labels
            k_range: Range of k values to test (default: 1, 3, 5, ..., 15)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k_range = k_range
        self.results = defaultdict(list)
        self.best_k = None
        self.best_score = None
        
    def run_cross_validation(self, n_runs=10, n_folds=10):
        """
        Perform multiple runs of k-fold cross-validation for different k values.
        
        Args:
            n_runs: Number of times to repeat cross-validation
            n_folds: Number of folds for cross-validation
        """
        print(f"Starting {n_runs} runs of {n_folds}-fold cross-validation")
        print(f"Testing k values: {list(self.k_range)}")
        
        start_time = time.time()
        
        for run in range(n_runs):
            seed = 42 + run  # Different seed for each run
            print(f"\nRun {run + 1}/{n_runs} (seed: {seed})")
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
            # Store scores for this run
            run_scores = defaultdict(list)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
                # Split data for this fold
                X_train_fold = self.X_train[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                # Test each k value
                for k in self.k_range:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_fold, y_train_fold)
                    score = knn.score(X_val_fold, y_val_fold)
                    run_scores[k].append(score)
            
            # Calculate mean scores for this run
            for k in self.k_range:
                mean_score = np.mean(run_scores[k])
                self.results[k].append(mean_score)
                print(f"k={k}: {mean_score:.4f}")
        
        self.analyze_results()
        
        total_time = time.time() - start_time
        print(f"\nTotal optimization time: {total_time:.2f} seconds")
    
    def analyze_results(self):
        """
        Analyze cross-validation results and determine the best k value.
        """
        # Calculate mean and standard deviation for each k
        stats = {}
        for k in self.k_range:
            scores = self.results[k]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            stats[k] = {
                'mean': mean_score,
                'std': std_score,
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        # Find best k value based on mean score
        self.best_k = max(stats.keys(), key=lambda k: stats[k]['mean'])
        self.best_score = stats[self.best_k]['mean']
        
        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 50)
        print("k\tMean ± Std\t\tMin\t\tMax")
        print("-" * 50)
        for k in self.k_range:
            stat = stats[k]
            print(f"{k}\t{stat['mean']:.4f} ± {stat['std']:.4f}\t"
                  f"{stat['min']:.4f}\t{stat['max']:.4f}")
        
        print("\nBest Results:")
        print(f"Best k: {self.best_k}")
        print(f"Mean accuracy: {self.best_score:.4f}")
        print(f"Standard deviation: {stats[self.best_k]['std']:.4f}")
    
    def get_best_classifier(self):
        """
        Return a KNN classifier initialized with the best k value.
        """
        if self.best_k is None:
            raise ValueError("Must run cross-validation before getting best classifier")
        return KNeighborsClassifier(n_neighbors=self.best_k)