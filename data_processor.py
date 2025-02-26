import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class BallisticsDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_dataset(self):
        """
        Load and preprocess the ballistics dataset
        """
        images = []
        labels = []
        
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(image_path)
                        if img is None:
                            continue
                        img = cv2.resize(img, (224, 224))
                        
                        images.append(img)
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def split_dataset(self, X, y, test_size=0.2):
        """
        Split dataset into training and testing sets
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def augment_data(self, image):
        """
        Perform data augmentation on single image
        """
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Rotate image
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated)
        
        # Flip image
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Adjust brightness
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        augmented_images.append(bright)
        
        return augmented_images 