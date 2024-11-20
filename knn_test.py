#Cross Validation test for KNN
from mnist_loader import MNISTLoader
from knn_optimizer import KNNOptimizer

def optimize_knn(sample_size=None):
    """
    Load data and perform KNN optimization.
    
    Args:
        sample_size: Optional number of samples to use (for faster testing)
    """
    # Load data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Use subset of data if specified
    if sample_size is not None:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Using {sample_size} samples for optimization")
    
    # Create and run optimizer
    optimizer = KNNOptimizer(X_train, y_train)
    optimizer.run_cross_validation()
    
    # Get best classifier and evaluate on test set
    best_knn = optimizer.get_best_classifier()
    best_knn.fit(X_train, y_train)
    test_score = best_knn.score(X_test, y_test)
    
    print("\nFinal Evaluation:")
    print(f"Test set accuracy with best k={optimizer.best_k}: {test_score:.4f}")
    
    return optimizer, best_knn

if __name__ == "__main__":
    # Use a smaller sample size for faster testing
    # Remove sample_size parameter or increase it for full optimization
    optimizer, best_knn = optimize_knn()