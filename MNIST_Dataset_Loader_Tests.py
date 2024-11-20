#MNIST_Dataset_Loader_Tests.py
import numpy as np
from mnist_loader import MNISTLoader

def display_normalization_examples(loader, num_examples=5):
    """
    Display original and normalized values for sample images.
    """
    print("\nNormalization Examples:")
    print("-" * 50)
    
    # Get raw data before normalization
    raw_train_images = loader.train_images
    
    # Get normalized and flattened data
    train_normalized, _, _, _ = loader.load_data()
    
    # Select random indices
    random_indices = np.random.choice(len(raw_train_images), num_examples, replace=False)
    
    for idx in random_indices:
        print(f"\nExample {idx} (Label: {loader.train_labels[idx]}):")
        print("\nOriginal values (sample of 25 pixels):")
        # Show a 5x5 sample of pixels from the original image
        sample_original = raw_train_images[idx, 10:15, 10:15]
        print(sample_original)
        
        print("\nNormalized values (same pixels):")
        # Show the same pixels from the normalized image
        # Reshape back to 2D to get the same pixels
        normalized_2d = train_normalized[idx].reshape(28, 28)
        sample_normalized = normalized_2d[10:15, 10:15]
        print(sample_normalized)
        
        print("\nValue ranges:")
        print(f"Original - Min: {raw_train_images[idx].min()}, Max: {raw_train_images[idx].max()}")
        print(f"Normalized - Min: {train_normalized[idx].min():.4f}, Max: {train_normalized[idx].max():.4f}")
        print("-" * 50)

def test_mnist_loader():
    """
    Test the MNISTLoader class functionality.
    """
    # Initialize loader
    loader = MNISTLoader()
    
    # Load and process data
    train_images, train_labels, test_images, test_labels = loader.load_data()
    
    # Test 1: Check shapes
    assert train_images.shape[0] == 60000, "Wrong number of training samples"
    assert test_images.shape[0] == 10000, "Wrong number of test samples"
    assert train_images.shape[1] == 784, "Wrong dimension for flattened training images"
    assert test_images.shape[1] == 784, "Wrong dimension for flattened test images"
    
    # Test 2: Check normalization
    assert np.all(train_images >= 0) and np.all(train_images <= 1), "Training images not normalized to [0, 1]"
    assert np.all(test_images >= 0) and np.all(test_images <= 1), "Test images not normalized to [0, 1]"
    
    # Test 3: Check labels
    assert train_labels.shape[0] == 60000, "Wrong number of training labels"
    assert test_labels.shape[0] == 10000, "Wrong number of test labels"
    assert np.all(train_labels >= 0) and np.all(train_labels <= 9), "Invalid training labels"
    assert np.all(test_labels >= 0) and np.all(test_labels <= 9), "Invalid test labels"
    
    print("All tests passed!")
    
    # Display normalization examples
    display_normalization_examples(loader)

if __name__ == "__main__":
    try:
        test_mnist_loader()
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")