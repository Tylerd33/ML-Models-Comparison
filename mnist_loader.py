#mnist_loader.py
import numpy as np

def read_idx_file(filename):
    """
    Read IDX file format used by MNIST dataset.
    """
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        n_dims = magic % 256
        
        # Read dimension sizes
        dims = []
        for i in range(n_dims):
            dims.append(int.from_bytes(f.read(4), 'big'))
            
        # Read data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Reshape based on dimensions
        data = data.reshape(dims)
        
        return data

class MNISTLoader:
    def __init__(self, data_path='.'):
        """
        Initialize MNIST loader with path to dataset files.
        
        Args:
            data_path (str): Directory containing MNIST files
        """
        self.data_path = data_path
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
    def load_data(self):
        """
        Load MNIST dataset and normalize images to [0, 1] range.
        Returns normalized and flattened data.
        """
        # Load raw data
        self.train_images = read_idx_file(f"{self.data_path}/train-images.idx3-ubyte")
        self.train_labels = read_idx_file(f"{self.data_path}/train-labels.idx1-ubyte")
        self.test_images = read_idx_file(f"{self.data_path}/t10k-images.idx3-ubyte")
        self.test_labels = read_idx_file(f"{self.data_path}/t10k-labels.idx1-ubyte")
        
        # Normalize images to [0, 1]
        train_images_normalized = self.train_images.astype('float32') / 255.0
        test_images_normalized = self.test_images.astype('float32') / 255.0
        
        # Flatten images to 784-dimensional vectors
        train_images_flattened = train_images_normalized.reshape(train_images_normalized.shape[0], -1)
        test_images_flattened = test_images_normalized.reshape(test_images_normalized.shape[0], -1)
        
        return (train_images_flattened, self.train_labels,
                test_images_flattened, self.test_labels)
    
    def get_original_shapes(self):
        """
        Return original shapes of the datasets.
        """
        return {
            'train_images': self.train_images.shape if self.train_images is not None else None,
            'train_labels': self.train_labels.shape if self.train_labels is not None else None,
            'test_images': self.test_images.shape if self.test_images is not None else None,
            'test_labels': self.test_labels.shape if self.test_labels is not None else None
        }