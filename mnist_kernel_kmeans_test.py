from mnist_loader import MNISTLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from mnist_kernel_kmeans import KernelKMeans

def test_kernel_kmeans(sample_size=None):
    """
    Test kernel k-means on MNIST dataset.
    
    Args:
        sample_size: Optional number of samples to use
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
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different kernels
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    
    for kernel in kernels:
        print(f"\nTesting {kernel} kernel...")
        
        # Initialize and train model
        model = KernelKMeans(
            n_clusters=10,
            kernel=kernel,
            max_iter=100,
            random_state=42
        )
        
        start_time = time.time()
        model.fit(X_train_scaled)
        training_time = time.time() - start_time
        
        # Evaluate
        train_labels = model.labels_
        test_labels = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_silhouette = silhouette_score(X_train_scaled, train_labels)
        train_ari = adjusted_rand_score(y_train, train_labels)
        test_ari = adjusted_rand_score(y_test, test_labels)
        
        results[kernel] = {
            'training_time': training_time,
            'train_silhouette': train_silhouette,
            'train_ari': train_ari,
            'test_ari': test_ari,
            'n_iterations': model.n_iter_
        }
        
        print(f"\nResults for {kernel} kernel:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Number of iterations: {model.n_iter_}")
        print(f"Training Silhouette Score: {train_silhouette:.4f}")
        print(f"Training Adjusted Rand Index: {train_ari:.4f}")
        print(f"Test Adjusted Rand Index: {test_ari:.4f}")
    
    # Plot comparison
    plot_kernel_comparison(results)
    
    return results

def plot_kernel_comparison(results):
    """
    Plot comparison of different kernels.
    """
    metrics = ['training_time', 'train_silhouette', 'train_ari', 'test_ari']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results[k][metric] for k in results]
        axes[i].bar(results.keys(), values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Value')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use a smaller sample size for faster testing
    results = test_kernel_kmeans(sample_size=5000)