from mnist_loader import MNISTLoader
from svm_optimizer import SVMOptimizer
import numpy as np

def optimize_svm(sample_size=None):
    """
    Load data and perform SVM optimization.
    
    Args:
        sample_size: Optional number of samples to use (for faster testing)
    """
    # Load data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Create optimizer
    optimizer = SVMOptimizer(X_train, y_train)
    
    # Run grid search
    optimizer.run_grid_search(sample_size=sample_size)
    
    # Validate best model
    optimizer.validate_best_model()
    
    # Print detailed results
    optimizer.print_detailed_results()
    
    # Get best model and evaluate on test set
    best_svm = optimizer.get_best_model()
    
    # If we used a sample for optimization, train the best model on full dataset
    if sample_size is not None:
        print("\nTraining best model configuration on full dataset...")
        best_svm.fit(X_train, y_train)
    
    test_score = best_svm.score(X_test, y_test)
    
    print("\nFinal Evaluation:")
    print(f"Best parameters: {optimizer.best_params}")
    print(f"Test set accuracy: {test_score:.4f}")
    
    return optimizer, best_svm

if __name__ == "__main__":
    # Use a smaller sample size for faster testing
    # Remove sample_size parameter or increase it for full optimization
    optimizer, best_svm = optimize_svm()