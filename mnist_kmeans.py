from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment


class MNISTClusterer:
    def __init__(self, n_clusters=10, random_state=42):
        """Initialize the clusterer with parameters."""
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10  # Changed back to 10 to match _compute_kmeans
        )
        self.pca = PCA(n_components=2)
        self.cluster_centers_ = None
        self.labels_ = None
        self.reduced_centers = None
        self.reduced_data = None
        self.elbow_results = None
        self.cluster_to_digit_map = None
        self.X_fit_ = None  # Store training data for prediction

    def _map_clusters_to_digits(self, y_true):
        """
        Map cluster labels to actual digits using the Hungarian algorithm.
        
        Args:
            y_true: True labels (must have same length as self.labels_)
        """
        if len(self.labels_) != len(y_true):
            raise ValueError(f"Length mismatch: labels_ has length {len(self.labels_)}, "
                           f"but y_true has length {len(y_true)}")
            
        n_clusters = self.n_clusters
        unique_digits = np.unique(y_true)
        n_digits = len(unique_digits)
        
        if n_clusters != n_digits:
            print(f"Warning: number of clusters ({n_clusters}) does not match "
                  f"number of unique digits ({n_digits})")
        
        # Create cost matrix
        max_clusters = max(n_clusters, n_digits)
        cost_matrix = np.zeros((max_clusters, max_clusters))
        
        # Count occurrences of each digit in each cluster
        for i in range(n_clusters):
            cluster_mask = (self.labels_ == i)
            for j, digit in enumerate(unique_digits):
                # Count how many times digit appears in cluster i
                digit_count = np.sum(y_true[cluster_mask] == digit)
                cost_matrix[i][j] = -digit_count
        
        # If we have more clusters than digits, pad the cost matrix
        if n_clusters > n_digits:
            # Pad with zeros (neutral cost)
            pass
        elif n_clusters < n_digits:
            # Pad with high cost to avoid mapping to these extra rows
            cost_matrix[n_clusters:, :] = 0
            
        # Use Hungarian algorithm to find optimal mapping
        cluster_ind, digit_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping dictionary
        mapping = {}
        for cluster, digit_idx in zip(cluster_ind, digit_ind):
            if cluster < n_clusters:  # Only map actual clusters
                mapping[cluster] = unique_digits[digit_idx] if digit_idx < len(unique_digits) else -1
                
        # Ensure all clusters have a mapping
        for i in range(n_clusters):
            if i not in mapping:
                # Assign to least used digit
                digit_counts = {d: 0 for d in unique_digits}
                for d in mapping.values():
                    if d in digit_counts:
                        digit_counts[d] += 1
                least_used_digit = min(digit_counts.items(), key=lambda x: x[1])[0]
                mapping[i] = least_used_digit
        
        return mapping
    def evaluate(self, X, y_true):
        """
        Evaluate clustering performance using various metrics.
        """
        try:
            if self.cluster_centers_ is None:
                self.fit(X)
            
            # Ensure X and y_true have the same length
            if len(X) != len(y_true):
                raise ValueError(f"Length mismatch: X has length {len(X)}, "
                               f"but y_true has length {len(y_true)}")
                
            # Get cluster assignments for this specific data
            self.labels_ = self.kmeans.predict(X)
            
            # Map clusters to digits
            self.cluster_to_digit_map = self._map_clusters_to_digits(y_true)
            
            # Verify all cluster labels have a mapping
            unique_clusters = np.unique(self.labels_)
            for cluster in unique_clusters:
                if cluster not in self.cluster_to_digit_map:
                    raise ValueError(f"Cluster {cluster} has no digit mapping")
            
            predicted_labels = np.array([self.cluster_to_digit_map[label] for label in self.labels_])
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predicted_labels)
            ari_score = adjusted_rand_score(y_true, self.labels_)
            nmi_score = normalized_mutual_info_score(y_true, self.labels_)
            silhouette = silhouette_score(X, self.labels_)
            
            # Get detailed classification report
            report = classification_report(y_true, predicted_labels)
            
            results = {
                'accuracy': accuracy,
                'adjusted_rand_index': ari_score,
                'normalized_mutual_info': nmi_score,
                'silhouette_score': silhouette,
                'classification_report': report
            }
            
            # Print results
            print("\nClustering Evaluation Metrics:")
            print("-" * 50)
            print(f"Accuracy (after mapping): {accuracy:.4f}")
            print(f"Adjusted Rand Index: {ari_score:.4f}")
            print(f"Normalized Mutual Information: {nmi_score:.4f}")
            print(f"Silhouette Score: {silhouette:.4f}")
            print("\nClassification Report:")
            print(report)
            
            # Print cluster to digit mapping
            print("\nCluster to Digit Mapping:")
            for cluster, digit in sorted(self.cluster_to_digit_map.items()):
                print(f"Cluster {cluster} → Digit {digit}")
                
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict digit labels for new data.
        """
        if self.cluster_to_digit_map is None:
            raise ValueError("Must run evaluate() before making predictions")
            
        try:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
                
            cluster_labels = self.kmeans.predict(X)
            predicted_digits = np.array([self.cluster_to_digit_map[label] for label in cluster_labels])
            return predicted_digits
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise


    def _compute_kmeans(self, params):
        """
        Helper function to compute k-means for parallel processing.
        """
        k, X = params
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, kmeans.labels_)
        return k, inertia, silhouette
        
    def find_optimal_k(self, X, k_range=range(2, 21)):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Input data
            k_range: Range of k values to test
        """
        print("Finding optimal number of clusters...")
        
        # Prepare parameters for parallel processing
        params = [(k, X) for k in k_range]
        
        # Run k-means for different k values in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._compute_kmeans, params))
            
        # Organize results
        k_values = []
        inertias = []
        silhouette_scores = []
        
        for k, inertia, silhouette in sorted(results):
            k_values.append(k)
            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            
        self.elbow_results = {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
        # Find optimal k using the elbow method
        optimal_k = self._find_elbow_point(k_values, inertias)
        
        # Find k with highest silhouette score
        best_silhouette_k = k_values[np.argmax(silhouette_scores)]
        
        return optimal_k, best_silhouette_k
    
    def _find_elbow_point(self, k_values, inertias):
        """
        Find the elbow point using the maximum curvature method.
        """
        coords = np.column_stack([k_values, inertias])
        
        # Normalize coordinates
        coords_normalized = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        
        # Calculate angles
        vectors = np.diff(coords_normalized, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        curvature = np.diff(angles)
        
        # Find point of maximum curvature
        elbow_idx = np.argmax(np.abs(curvature)) + 1
        return k_values[elbow_idx]
    
    def plot_elbow_analysis(self):
        """
        Plot elbow method analysis results.
        """
        if self.elbow_results is None:
            raise ValueError("Must run find_optimal_k first")
            
        k_values = self.elbow_results['k_values']
        inertias = self.elbow_results['inertias']
        silhouette_scores = self.elbow_results['silhouette_scores']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Elbow curve
        ax1.plot(k_values, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        
        # Add vertical line at optimal k
        optimal_k = self._find_elbow_point(k_values, inertias)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', 
                   label=f'Elbow Point (k={optimal_k})')
        ax1.legend()
        
        # Plot 2: Silhouette scores
        ax2.plot(k_values, silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        
        # Add vertical line at best silhouette score
        best_k = k_values[np.argmax(silhouette_scores)]
        ax2.axvline(x=best_k, color='r', linestyle='--',
                   label=f'Best Score (k={best_k})')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return optimal_k, best_k
    def fit(self, X):
        """Fit K-means clustering to the data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
            
        # Store training data
        self.X_fit_ = X.copy()
            
        print("Fitting K-means clustering...")
        try:
            self.labels_ = self.kmeans.fit_predict(X)
            self.cluster_centers_ = self.kmeans.cluster_centers_
            
            # Compute silhouette score
            if len(np.unique(self.labels_)) > 1:  # Check if we have more than one cluster
                score = silhouette_score(X, self.labels_)
                print(f"Silhouette score: {score:.4f}")
            
            # Reduce dimensionality for visualization
            print("Reducing dimensionality with PCA...")
            self.reduced_data = self.pca.fit_transform(X)
            self.reduced_centers = self.pca.transform(self.cluster_centers_)
            
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            raise
            
        return self

    def analyze_clusters(self, y_true):
        """
        Analyze cluster composition with respect to true labels.
        
        Args:
            y_true: True labels for the data
        """
        print("\nAnalyzing cluster composition...")
        cluster_compositions = defaultdict(lambda: defaultdict(int))
        
        for cluster_label, true_label in zip(self.labels_, y_true):
            cluster_compositions[cluster_label][true_label] += 1
            
        print("\nCluster composition analysis:")
        print("-" * 50)
        for cluster in range(self.n_clusters):
            total = sum(cluster_compositions[cluster].values())
            print(f"\nCluster {cluster}:")
            print(f"Total points: {total}")
            print("Digit distribution:")
            for digit, count in sorted(cluster_compositions[cluster].items()):
                percentage = (count / total) * 100
                print(f"Digit {digit}: {count} ({percentage:.1f}%)")
    
    def plot_clusters(self, y_true=None):
        """
        Create visualizations of the clusters.
        """
        if self.reduced_data is None or self.labels_ is None:
            raise ValueError("Must fit the model before plotting")
            
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 8))
            
            # Plot 1: Clusters with assigned labels
            ax1 = fig.add_subplot(121)
            
            # Get the data for plotting and ensure all arrays have matching lengths
            plot_data = self.reduced_data
            plot_labels = self.labels_
            
            if len(plot_data) != len(plot_labels):
                min_len = min(len(plot_data), len(plot_labels))
                plot_data = plot_data[:min_len]
                plot_labels = plot_labels[:min_len]
                print(f"Warning: Truncating data to match length {min_len}")
            
            # Create scatter plot with cluster labels
            scatter = ax1.scatter(
                plot_data[:, 0],
                plot_data[:, 1],
                c=plot_labels,
                cmap='tab10',
                vmin=0,
                vmax=max(9, self.n_clusters-1)  # Ensure proper color range
            )
            
            # Plot centroids
            if self.reduced_centers is not None:
                ax1.scatter(
                    self.reduced_centers[:, 0],
                    self.reduced_centers[:, 1],
                    marker='x',
                    s=200,
                    linewidths=3,
                    color='black',
                    label='Centroids'
                )
            
            ax1.set_title('Clusters with Centroids')
            ax1.legend()
            plt.colorbar(scatter, ax=ax1, label='Cluster Label')
            
            # Plot 2: Points colored by true digits (if provided)
            if y_true is not None:
                ax2 = fig.add_subplot(122)
                
                # Ensure all arrays have matching lengths for true label plot
                if len(y_true) != len(plot_data):
                    min_len = min(len(y_true), len(plot_data))
                    true_plot_data = plot_data[:min_len]
                    true_labels = y_true[:min_len]
                    print(f"Warning: Truncating true labels to match length {min_len}")
                else:
                    true_plot_data = plot_data
                    true_labels = y_true
                
                scatter = ax2.scatter(
                    true_plot_data[:, 0],
                    true_plot_data[:, 1],
                    c=true_labels,
                    cmap='tab10',
                    vmin=0,
                    vmax=9  # MNIST has 10 digits (0-9)
                )
                ax2.set_title('True Digit Labels')
                plt.colorbar(scatter, ax=ax2, label='True Digit')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error during plotting: {str(e)}")
            plt.close()  # Clean up in case of error
        finally:
            plt.close()  # Ensure figure is closed properly
        
    def visualize_centers(self):
        """
        Visualize cluster centers as digit images.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Must fit the model before visualizing centers")
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Plot each cluster center
        for idx, center in enumerate(self.cluster_centers_):
            # Reshape center to 28x28 image
            center_image = center.reshape(28, 28)
            
            # Display image
            axes[idx].imshow(center_image, cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(f'Cluster {idx}')
            
        plt.tight_layout()
        plt.show()