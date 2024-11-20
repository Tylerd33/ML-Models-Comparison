from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np

class MNISTClassifiers:
    def __init__(self):
        """
        Initialize the classifiers with default parameters
        """
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.logistic = LogisticRegression(max_iter=1000, multi_class='ovr')
        self.svm = SVC(kernel='rbf', random_state=42)
        
        self.models = {
            'KNN': self.knn,
            'Logistic Regression': self.logistic,
            'SVM': self.svm
        }
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, sample_size=None):
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sample_size: Number of samples to use for training (for faster testing)
        """
        results = {}
        
        # Use subset of data if sample_size is specified
        if sample_size is not None:
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train the model
            model.fit(X_train_subset, y_train_subset)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get detailed classification report
            report = classification_report(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'report': report
            }
            
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)
            
        return results
    
    def predict(self, model_name, X):
        """
        Make predictions using a specific model.
        
        Args:
            model_name: Name of the model to use ('KNN', 'Logistic Regression', or 'SVM')
            X: Features to predict
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        return self.models[model_name].predict(X)