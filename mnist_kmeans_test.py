from mnist_loader import MNISTLoader
from mnist_kmeans import MNISTClusterer
import numpy as np
import time

def test_clustering(sample_size=None):
    """
    Load data, perform k-means clustering, and evaluate performance.
    
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
        test_indices = np.random.choice(len(X_test), sample_size//5, replace=False)
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        print(f"Using {sample_size} training samples and {sample_size//5} test samples")
    
    # Create clusterer (with arbitrary initial k, will be optimized)
    clusterer = MNISTClusterer(n_clusters=10)
    
    # Find optimal k
    print("\nAnalyzing optimal number of clusters...")
    optimal_k, best_silhouette_k = clusterer.find_optimal_k(X_train)
    
    # Plot elbow analysis
    print("\nGenerating elbow analysis plots...")
    clusterer.plot_elbow_analysis()
    
    print(f"\nElbow method suggests k = {optimal_k}")
    print(f"Best silhouette score achieved with k = {best_silhouette_k}")
    
    # Create new clusterer with optimal k
    print(f"\nTraining K-means with optimal k = {optimal_k}...")
    start_time = time.time()
    
    optimal_clusterer = MNISTClusterer(n_clusters=optimal_k)
    optimal_clusterer.fit(X_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on training set
    print("\nEvaluating on training set:")
    train_results = optimal_clusterer.evaluate(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_results = optimal_clusterer.evaluate(X_test, y_test)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    optimal_clusterer.plot_clusters(y_true=y_train)
    optimal_clusterer.visualize_centers()
    
    # Compare train and test performance
    print("\nPerformance Comparison:")
    print("-" * 50)
    metrics = ['accuracy', 'adjusted_rand_index', 'normalized_mutual_info', 'silhouette_score']
    print(f"{'Metric':<25} {'Train':>10} {'Test':>10}")
    print("-" * 50)
    for metric in metrics:
        train_value = train_results[metric]
        test_value = test_results[metric]
        print(f"{metric:<25} {train_value:>10.4f} {test_value:>10.4f}")
    
    return optimal_clusterer, train_results, test_results

if __name__ == "__main__":
    # Use a smaller sample size for faster testing
    # Remove sample_size parameter or increase it for full clustering
    clusterer, train_results, test_results = test_clustering()



