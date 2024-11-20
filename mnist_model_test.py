from mnist_loader import MNISTLoader
from mnist_models import MNISTClassifiers
import numpy as np

def train_and_evaluate_models(sample_size=None):
    """
    Load data, train models, and evaluate their performance.
    
    Args:
        sample_size: Optional number of training samples to use (for faster testing)
    """
    # Load and prepare data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    print(f"\nData shapes:")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Initialize classifiers
    classifiers = MNISTClassifiers()
    
    # Train and evaluate all models
    results = classifiers.train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        sample_size=sample_size
    )
    
    return classifiers, results

def test_specific_samples(classifiers, loader, num_samples=5):
    """
    Test models on specific samples and compare their predictions.
    """
    # Load data
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Select random test samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    print("\nTesting specific samples:")
    print("-" * 50)
    
    for idx in indices:
        X_sample = X_test[idx:idx+1]  # Keep 2D shape for sklearn
        true_label = y_test[idx]
        
        print(f"\nSample {idx} (True label: {true_label})")
        print("Predictions:")
        
        for model_name in classifiers.models.keys():
            prediction = classifiers.predict(model_name, X_sample)[0]
            print(f"{model_name}: {prediction}")
            
        print("-" * 25)

if __name__ == "__main__":
    # Train models on a smaller subset for faster testing
    # Remove sample_size parameter or increase it for full training
    classifiers, results = train_and_evaluate_models()
    
    # Load data for specific sample testing
    loader = MNISTLoader()
    
    # Test specific samples
    test_specific_samples(classifiers, loader)