#Cross Validation optimizer for SVM
from mnist_loader import MNISTLoader
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
import numpy as np
import time
from collections import defaultdict

class SVMOptimizer:
    def __init__(self, X_train, y_train):
        """
        Initialize SVM optimizer.
        
        Args:
            X_train: Training features
            y_train: Training labelsx
        """
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.results = defaultdict(dict)
        
    def run_grid_search(self, sample_size=None):
        """
        Perform grid search for SVM parameters using cross-validation.
        
        Args:
            sample_size: Optional number of samples to use for faster search
        """
        # Use subset of data if specified
        if sample_size is not None:
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_subset = self.X_train[indices]
            y_subset = self.y_train[indices]
        else:
            X_subset = self.X_train
            y_subset = self.y_train
            
        print(f"Running grid search with {len(X_subset)} samples")
        
        # Define parameter grid
        param_grid = {
            'linear': {
                'kernel': ['linear'],
                'C': [0.1, 1, 10]
            },
            'rbf': {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'poly': {
                'kernel': ['poly'],
                'C': [0.1, 1, 10],
                'degree': [2, 3],
                'gamma': ['scale', 'auto']
            }
        }
        
        start_time = time.time()
        
        # Test each kernel type separately
        for kernel_type, params in param_grid.items():
            print(f"\nTesting {kernel_type} kernel configurations...")
            
            # Create grid search
            grid_search = GridSearchCV(
                SVC(random_state=42),
                param_grid=params,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy'
            )
            
            # Fit grid search
            grid_search.fit(X_subset, y_subset)
            
            # Store results
            self.results[kernel_type] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"\n{kernel_type} kernel results:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Update best overall model if necessary
            if (self.best_score is None or 
                grid_search.best_score_ > self.best_score):
                self.best_score = grid_search.best_score_
                self.best_params = grid_search.best_params_
                self.best_model = grid_search.best_estimator_
        
        total_time = time.time() - start_time
        print(f"\nTotal optimization time: {total_time:.2f} seconds")
        
    def validate_best_model(self, n_splits=5):
        """
        Perform additional cross-validation on the best model configuration.
        
        Args:
            n_splits: Number of cross-validation splits
        """
        if self.best_model is None:
            raise ValueError("Must run grid search before validation")
            
        print("\nValidating best model configuration...")
        scores = cross_val_score(
            self.best_model, 
            self.X_train, 
            self.y_train, 
            cv=n_splits,
            n_jobs=-1
        )
        
        print("\nCross-validation results for best model:")
        print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"Min accuracy: {scores.min():.4f}")
        print(f"Max accuracy: {scores.max():.4f}")
        
    def get_best_model(self):
        """
        Return the best SVM model found during optimization.
        """
        if self.best_model is None:
            raise ValueError("Must run grid search before getting best model")
        return self.best_model

    def print_detailed_results(self):
        """
        Print detailed results for all tested configurations.
        """
        print("\nDetailed Results by Kernel Type:")
        print("-" * 60)
        
        for kernel_type, results in self.results.items():
            print(f"\n{kernel_type.upper()} KERNEL:")
            print("-" * 30)
            
            cv_results = results['cv_results']
            params = cv_results['params']
            mean_scores = cv_results['mean_test_score']
            std_scores = cv_results['std_test_score']
            
            for i in range(len(params)):
                print(f"\nParameters: {params[i]}")
                print(f"Mean accuracy: {mean_scores[i]:.4f} ± {std_scores[i]:.4f}")