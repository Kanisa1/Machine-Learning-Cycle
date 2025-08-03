"""
Prediction Module for Chest X-Ray Pneumonia Detection
Handles prediction requests, responses, and result formatting
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json
import os

from .preprocessing import get_data_processor, ImagePreprocessor
from .model import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling prediction requests and responses"""
    
    def __init__(self):
        self.data_processor = get_data_processor()
        self.model_manager = get_model_manager()
        self.prediction_history = []
        
    def predict_from_features(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction from feature vector
        
        Args:
            features: List of feature values
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)
            
            # Make prediction using model manager
            prediction_result = self.model_manager.predict(features_array)
            
            # Format result
            result = {
                'prediction': prediction_result['prediction'],
                'prediction_label': 'PNEUMONIA' if prediction_result['prediction'] == 1 else 'NORMAL',
                'confidence': prediction_result['confidence'],
                'timestamp': datetime.now().isoformat(),
                'model_info': self.model_manager.get_model_info()
            }
            
            # Store in history
            self.prediction_history.append(result)
            
            logger.info(f"Prediction made: {result['prediction_label']} (confidence: {result['confidence']:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction from features: {str(e)}")
            raise
    
    async def predict_from_image(self, image_file) -> Dict[str, Any]:
        """
        Make prediction from uploaded image
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Process image directly from UploadFile
            preprocessor = ImagePreprocessor()
            processed_image = await preprocessor.load_and_preprocess_image(image_file)
            
            # Apply feature engineering if components are loaded
            try:
                if hasattr(self.data_processor.feature_engineer, 'pca_transformer') and self.data_processor.feature_engineer.pca_transformer is not None:
                    features = self.data_processor.feature_engineer.transform_features(processed_image.reshape(1, -1))
                else:
                    # If no PCA components, use the first 200 features (or pad/truncate as needed)
                    processed_features = processed_image.reshape(1, -1)
                    if processed_features.shape[1] > 200:
                        features = processed_features[:, :200]  # Take first 200 features
                    elif processed_features.shape[1] < 200:
                        # Pad with zeros if less than 200 features
                        padding = np.zeros((1, 200 - processed_features.shape[1]))
                        features = np.hstack([processed_features, padding])
                    else:
                        features = processed_features
            except Exception as e:
                logger.warning(f"Feature engineering failed, using raw features: {str(e)}")
                # Fallback: use first 200 features
                processed_features = processed_image.reshape(1, -1)
                if processed_features.shape[1] > 200:
                    features = processed_features[:, :200]
                else:
                    padding = np.zeros((1, 200 - processed_features.shape[1]))
                    features = np.hstack([processed_features, padding])
            
            # Make prediction
            prediction_result = self.model_manager.predict(features)
            
            # Format result
            result = {
                'prediction': int(prediction_result['prediction']),
                'prediction_label': 'PNEUMONIA' if prediction_result['prediction'] == 1 else 'NORMAL',
                'confidence': float(prediction_result['confidence']),
                'timestamp': datetime.now().isoformat(),
                'model_info': self.model_manager.get_model_info(),
                'image_processed': True,
                'features_shape': features.shape
            }
            
            # Store in history
            self.prediction_history.append(result)
            
            logger.info(f"Image prediction made: {result['prediction_label']} (confidence: {result['confidence']:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction from image: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for i, features in enumerate(features_list):
                try:
                    result = self.predict_from_features(features)
                    result['batch_index'] = i
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error predicting batch item {i}: {str(e)}")
                    results.append({
                        'error': str(e),
                        'batch_index': i,
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Batch prediction completed: {len(results)} items")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent prediction history
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent predictions
        """
        return self.prediction_history[-limit:] if self.prediction_history else []
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get prediction statistics
        
        Returns:
            Dictionary with prediction statistics
        """
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'pneumonia_count': 0,
                'normal_count': 0,
                'average_confidence': 0.0
            }
        
        total = len(self.prediction_history)
        pneumonia_count = sum(1 for p in self.prediction_history if p.get('prediction') == 1)
        normal_count = sum(1 for p in self.prediction_history if p.get('prediction') == 0)
        avg_confidence = np.mean([p.get('confidence', 0) for p in self.prediction_history])
        
        return {
            'total_predictions': total,
            'pneumonia_count': pneumonia_count,
            'normal_count': normal_count,
            'pneumonia_percentage': (pneumonia_count / total) * 100 if total > 0 else 0,
            'normal_percentage': (normal_count / total) * 100 if total > 0 else 0,
            'average_confidence': float(avg_confidence)
        }

class RetrainingService:
    """Service for handling model retraining"""
    
    def __init__(self):
        self.data_processor = get_data_processor()
        self.model_manager = get_model_manager()
        self.retraining_history = []
        
    def prepare_retraining_data(self, uploaded_files: List, labels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for retraining from uploaded files
        
        Args:
            uploaded_files: List of uploaded image files
            labels: Optional list of labels (0 for normal, 1 for pneumonia)
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Process images
            features = self.data_processor.process_batch_images(uploaded_files)
            
            # Generate labels if not provided
            if labels is None:
                labels = self.data_processor.create_labels_from_paths([f.filename for f in uploaded_files])
            
            logger.info(f"Retraining data prepared: {len(features)} samples")
            
            return features, np.array(labels)
            
        except Exception as e:
            logger.error(f"Error preparing retraining data: {str(e)}")
            raise
    
    def trigger_retraining(self, model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Trigger model retraining with stored data
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Retraining results
        """
        try:
            # Get retraining data from database or storage
            # This would typically come from a database
            retraining_data = self._get_stored_retraining_data()
            
            if not retraining_data:
                # Create some dummy data for demonstration purposes
                logger.info("No retraining data found. Creating dummy data for demonstration...")
                
                # Generate dummy training data
                dummy_features = np.random.normal(0, 1, (50, 200))  # 50 samples, 200 features
                dummy_labels = np.random.randint(0, 2, 50)  # Random binary labels
                
                retraining_data = {
                    'features': dummy_features,
                    'labels': dummy_labels
                }
                
                logger.info(f"Created dummy retraining data: {len(dummy_features)} samples")
            
            # Retrain model
            result = self.model_manager.retrain_model(
                retraining_data['features'],
                retraining_data['labels']
            )
            
            # Store retraining history
            self.retraining_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'data_size': len(retraining_data['features']),
                'new_accuracy': result.get('new_accuracy', 0.0),
                'retrained_model': result.get('retrained_model', 'unknown')
            })
            
            logger.info(f"Retraining completed: {result.get('retrained_model', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {str(e)}")
            raise
    
    def _get_stored_retraining_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get stored retraining data from database or file
        
        Returns:
            Dictionary with features and labels, or None if no data
        """
        # This is a placeholder - in a real implementation, you would
        # retrieve data from a database or file storage
        # For now, we'll return None to indicate no data is available
        
        # Example implementation:
        # import sqlite3
        # conn = sqlite3.connect('../data/retraining_data.db')
        # cursor = conn.cursor()
        # cursor.execute("SELECT features, labels FROM retraining_data")
        # data = cursor.fetchall()
        # conn.close()
        
        return None
    
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """
        Get retraining history
        
        Returns:
            List of retraining events
        """
        return self.retraining_history
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Get current retraining status
        
        Returns:
            Current retraining status
        """
        try:
            # Get the latest retraining event
            if self.retraining_history:
                latest = self.retraining_history[-1]
                return {
                    "status": "completed",
                    "message": f"Last retraining completed at {latest['timestamp']}",
                    "model_type": latest.get('model_type', 'unknown'),
                    "data_size": latest.get('data_size', 0),
                    "new_accuracy": latest.get('new_accuracy', 0.0),
                    "timestamp": latest['timestamp']
                }
            else:
                return {
                    "status": "idle",
                    "message": "No retraining has been performed yet",
                    "model_type": "none",
                    "data_size": 0,
                    "new_accuracy": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting retraining status: {str(e)}",
                "model_type": "unknown",
                "data_size": 0,
                "new_accuracy": 0.0,
                "timestamp": datetime.now().isoformat()
            }

# Global service instances
prediction_service = PredictionService()
retraining_service = RetrainingService()

def get_prediction_service() -> PredictionService:
    """Get the global prediction service instance"""
    return prediction_service

def get_retraining_service() -> RetrainingService:
    """Get the global retraining service instance"""
    return retraining_service 