"""
Data Preprocessing Module for Chest X-Ray Pneumonia Detection
Handles image processing, feature extraction, and data preparation
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import pickle
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for chest X-ray analysis"""
    
    def __init__(self, target_size: Tuple[int, int] = (150, 150)):
        self.target_size = target_size
        
    async def load_and_preprocess_image(self, image_path) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file or UploadFile object
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Handle UploadFile object
                image_data = await image_path.read()
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Flatten image
            image_flat = image.flatten()
            
            return image_flat
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of preprocessed images
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                processed_image = self.load_and_preprocess_image(image_path)
                processed_images.append(processed_image)
            except Exception as e:
                logger.warning(f"Skipping image {image_path}: {str(e)}")
                continue
        
        return np.array(processed_images)

class FeatureEngineer:
    """Handles feature engineering and dimensionality reduction"""
    
    def __init__(self, pca_components: int = 200):
        self.pca_components = pca_components
        self.pca_transformer = None
        self.feature_selector = None
        
    def load_preprocessing_components(self, components_path: str = None):
        """
        Load pre-trained preprocessing components
        
        Args:
            components_path: Path to the preprocessing components file
            
        Returns:
            True if successful, False otherwise
        """
        if components_path is None:
            # Try multiple possible paths
            possible_paths = [
                "models/preprocessing_components.pkl",
                "../models/preprocessing_components.pkl",
                "../../models/preprocessing_components.pkl",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "preprocessing_components.pkl")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    components_path = path
                    break
            else:
                raise FileNotFoundError(f"Could not find preprocessing_components.pkl in any of the expected locations: {possible_paths}")
            
        try:
            with open(components_path, 'rb') as f:
                components = pickle.load(f)
                self.pca_transformer = components.get('pca')
                self.feature_selector = components.get('feature_selector')
            logger.info(f"Preprocessing components loaded successfully from {components_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading preprocessing components from {components_path}: {str(e)}")
            raise
    
    def transform_features(self, images: np.ndarray) -> np.ndarray:
        """
        Transform raw images to PCA features
        
        Args:
            images: Raw image data (flattened)
            
        Returns:
            PCA-transformed features
        """
        if self.pca_transformer is None:
            raise ValueError("PCA transformer not loaded. Call load_preprocessing_components first.")
        
        try:
            # Apply PCA transformation
            features = self.pca_transformer.transform(images)
            logger.info(f"Transformed {images.shape[0]} images to {features.shape[1]} PCA features")
            return features
        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise
    
    def save_preprocessing_components(self, components_path: str = None):
        """
        Save preprocessing components for future use
        
        Args:
            components_path: Path to save the components
        """
        if components_path is None:
            # Get absolute path to models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(current_dir), "models")
            components_path = os.path.join(models_dir, "preprocessing_components.pkl")
            
        components = {
            'pca': self.pca_transformer,
            'feature_selector': self.feature_selector
        }
        
        try:
            with open(components_path, 'wb') as f:
                pickle.dump(components, f)
            logger.info(f"Preprocessing components saved to {components_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessing components: {str(e)}")
            raise

class DataProcessor:
    """Main data processing class that orchestrates preprocessing and feature engineering"""
    
    def __init__(self, target_size: Tuple[int, int] = (150, 150), pca_components: int = 200):
        self.image_preprocessor = ImagePreprocessor(target_size)
        self.feature_engineer = FeatureEngineer(pca_components)
        
    def process_single_image(self, image_path: str) -> np.ndarray:
        """
        Process a single image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed features ready for model prediction
        """
        # Load preprocessing components
        self.feature_engineer.load_preprocessing_components()
        
        # Preprocess image
        raw_image = self.image_preprocessor.load_and_preprocess_image(image_path)
        
        # Transform to features
        features = self.feature_engineer.transform_features(raw_image.reshape(1, -1))
        
        return features[0]  # Return single feature vector
    
    def process_batch_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Process a batch of images for training/retraining
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Processed features array
        """
        # Load preprocessing components
        self.feature_engineer.load_preprocessing_components()
        
        # Preprocess images
        raw_images = self.image_preprocessor.preprocess_batch(image_paths)
        
        # Transform to features
        features = self.feature_engineer.transform_features(raw_images)
        
        return features
    
    def create_labels_from_paths(self, image_paths: List[str]) -> np.ndarray:
        """
        Create labels based on image file paths
        Assumes directory structure: .../normal/... or .../pneumonia/...
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of labels (0 for normal, 1 for pneumonia)
        """
        labels = []
        
        for path in image_paths:
            if 'normal' in path.lower():
                labels.append(0)
            elif 'pneumonia' in path.lower():
                labels.append(1)
            else:
                # Default to normal if unclear
                labels.append(0)
        
        return np.array(labels)

# Global data processor instance
data_processor = DataProcessor()

def get_data_processor() -> DataProcessor:
    """Get the global data processor instance"""
    return data_processor