from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class KernelKMeans:
    """
    Kernel K-means implementation supporting various kernel functions.
    """
    VALID_KERNELS = {'rbf', 'poly', 'linear'}
    
    def __init__(self, n_clusters=10, kernel='rbf', max_iter=100, random_state=42, 
                 gamma='scale', degree=3, coef0=1, tol=1e-4):
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.max_iter = max_iter
        self.random_state = random_state
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        
        self.labels_ = None
        self.cluster_centers_indices_ = None
        self.n_iter_ = 0
        self.kernel_matrix = None
        self.inertia_ = None
        self.X_fit_ = None  # Store the training data

        
    def _validate_data(self, X):
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
            except:
                raise ValueError("X must be convertible to a numpy array")
                
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
            
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}")
            
        return X
        
    def _compute_kernel(self, X, Y=None):
        """
        Compute kernel matrix.
        """
        try:
            if self.gamma == 'scale' and Y is None:
                if X.var() == 0:
                    raise ValueError("Data has zero variance")
                self.gamma = 1.0 / (X.shape[1] * X.var())
                
            if self.kernel == 'rbf':
                return rbf_kernel(X, Y, gamma=self.gamma)
            elif self.kernel == 'poly':
                return polynomial_kernel(X, Y, degree=self.degree, 
                                      gamma=self.gamma, coef0=self.coef0)
            else:  # linear
                return linear_kernel(X, Y)
                
        except Exception as e:
            raise RuntimeError(f"Error computing kernel matrix: {str(e)}")
    
    def _init_centroids(self, n_samples):
        """
        Initialize cluster centers using k-means++ like initialization.
        """
        try:
            np.random.seed(self.random_state)
            
            # Choose first centroid randomly
            centroids = [np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Compute distances to existing centroids
                distances = np.inf * np.ones(n_samples)
                for j in range(len(centroids)):
                    current_dist = self.kernel_matrix[centroids[j], :]
                    distances = np.minimum(distances, current_dist)
                
                # Add small constant to avoid division by zero
                distances = np.maximum(distances, 1e-10)
                
                # Choose next centroid
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                next_centroid = np.random.choice(n_samples, p=probabilities)
                centroids.append(next_centroid)
            
            return np.array(centroids)
            
        except Exception as e:
            raise RuntimeError(f"Error initializing centroids: {str(e)}")
    
    def _compute_inertia(self, distances):
        """Compute clustering inertia (sum of squared distances)."""
        return np.sum(np.min(distances, axis=1))
    
    def fit(self, X):
        """
        Fit the kernel k-means model.
        
        Args:
            X: Training data
        """
        # Store the training data
        self.X_fit_ = X.copy()
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        print("Computing kernel matrix...")
        self.kernel_matrix = self._compute_kernel(X)
        
        # Initialize centroids
        print("Initializing centroids...")
        self.cluster_centers_indices_ = self._init_centroids(n_samples)
        self.labels_ = np.zeros(n_samples, dtype=int)
        
        # Initialize variables
        old_labels = None
        best_inertia = np.inf
        best_labels = None
        
        print("Training kernel k-means...")
        for n_iter in tqdm(range(self.max_iter)):
            # Compute distances to centroids
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                center_idx = self.cluster_centers_indices_[j]
                distances[:, j] = (self.kernel_matrix.diagonal() + 
                                 self.kernel_matrix[center_idx, center_idx] - 
                                 2 * self.kernel_matrix[:, center_idx])
            
            # Assign points to nearest centroid
            new_labels = np.argmin(distances, axis=1)
            
            # Compute inertia
            current_inertia = np.sum(np.min(distances, axis=1))
            
            # Update best solution
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_labels = new_labels.copy()
            
            # Check for convergence
            if old_labels is not None:
                change = np.mean(new_labels != old_labels)
                if change < self.tol:
                    break
            
            # Update centroids
            for j in range(self.n_clusters):
                cluster_points = np.where(new_labels == j)[0]
                if len(cluster_points) > 0:
                    within_distances = self.kernel_matrix[cluster_points][:, cluster_points]
                    centroid_idx = cluster_points[np.argmin(
                        within_distances.sum(axis=1) / len(cluster_points)
                    )]
                    self.cluster_centers_indices_[j] = centroid_idx
            
            old_labels = new_labels
        
        # Set final results
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = n_iter + 1
        
        print(f"Converged after {self.n_iter_} iterations")
        return self
    
    def predict(self, X_new):
        """
        Predict cluster labels for new data.
        
        Args:
            X_new: New data points
        """
        if self.cluster_centers_indices_ is None or self.X_fit_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
            
        try:
            # Get the center points from training data
            centers = self.X_fit_[self.cluster_centers_indices_]
            
            # Compute kernel between new points and center points
            kernel_new = self._compute_kernel(X_new, centers)
            
            # Compute self-similarity of new points
            self_similarity = np.diag(self._compute_kernel(X_new))
            
            # Compute self-similarity of centers
            center_self_similarity = np.diag(self._compute_kernel(centers))
            
            # Compute distances to centroids in feature space
            distances = (self_similarity[:, np.newaxis] + 
                       center_self_similarity - 
                       2 * kernel_new)
            
            return np.argmin(distances, axis=1)
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")