"""
Model Management Module for Chest X-Ray Pneumonia Detection
Handles model loading, training, evaluation, and retraining
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML models for chest X-ray pneumonia detection"""
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Get absolute path to models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(os.path.dirname(current_dir), "models")
        else:
            self.models_dir = models_dir
        self.current_model = None
        self.model_info = {}
        self.retraining_data = []
        
    def load_model(self, model_name: str = "logistic_regression_preprocessed_model.pkl") -> Any:
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Loaded model object
        """
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.current_model = model
            logger.info(f"Model loaded successfully: {model_name}")
            
            # Store model information
            self.model_info = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'loaded_at': datetime.now().isoformat()
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def save_model(self, model: Any, model_name: str) -> bool:
        """
        Save a model to disk
        
        Args:
            model: Model object to save
            model_name: Name for the saved model file
            
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(self.models_dir, model_name)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved successfully: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.current_model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_loaded": True,
            "model_info": self.model_info,
            "model_type": type(self.current_model).__name__,
            "models_directory": self.models_dir
        }
    
    def list_available_models(self) -> List[str]:
        """List all available model files"""
        try:
            if not os.path.exists(self.models_dir):
                return []
            
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            return model_files
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using the current model
        
        Args:
            features: Feature array for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        try:
            # Make prediction
            prediction = self.current_model.predict(features.reshape(1, -1))[0]
            
            # Get prediction probabilities if available
            if hasattr(self.current_model, 'predict_proba'):
                probabilities = self.current_model.predict_proba(features.reshape(1, -1))[0]
                confidence = max(probabilities)
                pneumonia_prob = probabilities[1] if len(probabilities) > 1 else 0.0
            else:
                confidence = 1.0 if prediction else 0.0
                pneumonia_prob = float(prediction)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'pneumonia_probability': float(pneumonia_prob),
                'normal_probability': float(1 - pneumonia_prob) if len(probabilities) > 1 else float(1 - prediction)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Train a new model
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to train
            
        Returns:
            Training results
        """
        try:
            # Select model based on type
            if model_type == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(random_state=42)
            elif model_type == "svm":
                model = SVC(probability=True, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on training data
            train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            
            # Store the trained model
            self.current_model = model
            
            # Save model
            model_name = f"{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.save_model(model, model_name)
            
            results = {
                'model_type': model_type,
                'model_name': model_name,
                'train_accuracy': train_accuracy,
                'training_samples': len(X_train),
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model training completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the current model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        try:
            # Make predictions
            y_pred = self.current_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'test_samples': len(X_test),
                'evaluated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model evaluation completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def retrain_model(self, new_X: np.ndarray, new_y: np.ndarray) -> Dict[str, Any]:
        """
        Retrain the current model with new data
        
        Args:
            new_X: New training features
            new_y: New training labels
            
        Returns:
            Retraining results
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        try:
            # Retrain the model
            self.current_model.fit(new_X, new_y)
            
            # Evaluate on new data
            new_pred = self.current_model.predict(new_X)
            new_accuracy = accuracy_score(new_y, new_pred)
            
            # Save retrained model
            model_name = f"retrained_{self.model_info.get('model_name', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.save_model(self.current_model, model_name)
            
            results = {
                'retrained_model': model_name,
                'new_accuracy': float(new_accuracy),
                'new_samples': len(new_X),
                'retrained_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model retraining completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            raise

# Global model manager instance
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    return model_manager