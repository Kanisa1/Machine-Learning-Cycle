"""
Locust Load Testing Script for Chest X-Ray Pneumonia Detection API
Simulates flood requests to test model performance and response times
"""

from locust import HttpUser, task, between, events
import random
import numpy as np
import json
import time

class MLPipelineUser(HttpUser):
    """Simulates user behavior for the ML Pipeline API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.features_sample = self.generate_sample_features()
        self.test_images = [
            "sample_normal.jpg",
            "sample_pneumonia.jpg"
        ]
    
    def generate_sample_features(self):
        """Generate sample PCA features for testing"""
        # Generate 200 PCA features (normal distribution)
        return np.random.normal(0, 1, 200).tolist()
    
    @task(3)
    def health_check(self):
        """Health check endpoint - most frequent request"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check returned unhealthy status")
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def single_prediction(self):
        """Single prediction from features"""
        features = self.generate_sample_features()
        
        with self.client.post("/predict", 
                             json=features,
                             catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid prediction response format")
            else:
                response.failure(f"Prediction failed with status {response.status_code}")
    
    @task(1)
    def batch_prediction(self):
        """Batch prediction with multiple feature vectors"""
        features_list = [self.generate_sample_features() for _ in range(random.randint(2, 5))]
        
        with self.client.post("/predict_batch",
                             json=features_list,
                             catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and "total_predictions" in data:
                    response.success()
                else:
                    response.failure("Invalid batch prediction response format")
            else:
                response.failure(f"Batch prediction failed with status {response.status_code}")
    
    @task(1)
    def model_info(self):
        """Get model information"""
        with self.client.get("/model_info", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "current_model" in data:
                    response.success()
                else:
                    response.failure("Invalid model info response format")
            else:
                response.failure(f"Model info failed with status {response.status_code}")
    
    @task(1)
    def prediction_history(self):
        """Get prediction history"""
        limit = random.randint(10, 50)
        
        with self.client.get(f"/prediction_history?limit={limit}", 
                            catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and "statistics" in data:
                    response.success()
                else:
                    response.failure("Invalid prediction history response format")
            else:
                response.failure(f"Prediction history failed with status {response.status_code}")
    
    @task(1)
    def visualizations(self):
        """Get data visualizations"""
        with self.client.get("/visualizations", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "feature_stories" in data and "prediction_statistics" in data:
                    response.success()
                else:
                    response.failure("Invalid visualizations response format")
            else:
                response.failure(f"Visualizations failed with status {response.status_code}")
    
    @task(1)
    def trigger_retraining(self):
        """Trigger model retraining"""
        # Note: This is a heavy operation, so we use it sparingly
        if random.random() < 0.1:  # Only 10% chance
            with self.client.post("/trigger_retraining",
                                 data={"model_type": "logistic_regression"},
                                 catch_response=True) as response:
                if response.status_code == 200:
                    data = response.json()
                    if "message" in data and "status" in data:
                        response.success()
                    else:
                        response.failure("Invalid retraining response format")
                else:
                    response.failure(f"Retraining failed with status {response.status_code}")

class HighLoadUser(HttpUser):
    """Simulates high-load scenarios with rapid requests"""
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        """Initialize high-load user"""
        self.features = np.random.normal(0, 1, 200).tolist()
    
    @task(5)
    def rapid_health_checks(self):
        """Rapid health check requests"""
        self.client.get("/health")
    
    @task(3)
    def rapid_predictions(self):
        """Rapid prediction requests"""
        self.client.post("/predict", json=self.features)
    
    @task(2)
    def rapid_model_info(self):
        """Rapid model info requests"""
        self.client.get("/model_info")

class StressTestUser(HttpUser):
    """Simulates stress testing scenarios"""
    
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Initialize stress test user"""
        self.large_batch = [np.random.normal(0, 1, 200).tolist() for _ in range(20)]
    
    @task(1)
    def large_batch_prediction(self):
        """Large batch prediction to stress the system"""
        with self.client.post("/predict_batch",
                             json=self.large_batch,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Large batch failed with status {response.status_code}")
    
    @task(1)
    def concurrent_operations(self):
        """Simulate concurrent operations"""
        # Make multiple requests in quick succession
        responses = []
        
        # Health check
        responses.append(self.client.get("/health"))
        
        # Single prediction
        features = np.random.normal(0, 1, 200).tolist()
        responses.append(self.client.post("/predict", json=features))
        
        # Model info
        responses.append(self.client.get("/model_info"))
        
        # Check if all succeeded
        success_count = sum(1 for r in responses if r.status_code == 200)
        if success_count == len(responses):
            return True
        else:
            return False

# Custom event listeners for monitoring
@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Custom request handler for detailed monitoring"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("Load testing started!")
    print(f"Target host: {environment.host}")
    print(f"Number of users: {environment.runner.user_count}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("Load testing completed!")
    
    # Print summary statistics
    stats = environment.stats
    print(f"\nTest Summary:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")

# Configuration for different test scenarios
class TestScenarios:
    """Predefined test scenarios for different load patterns"""
    
    @staticmethod
    def light_load():
        """Light load scenario - 5 users, 1-3 second intervals"""
        return {
            "users": 5,
            "spawn_rate": 1,
            "run_time": "2m"
        }
    
    @staticmethod
    def medium_load():
        """Medium load scenario - 15 users, 1-2 second intervals"""
        return {
            "users": 15,
            "spawn_rate": 2,
            "run_time": "5m"
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load scenario - 30 users, 0.5-1 second intervals"""
        return {
            "users": 30,
            "spawn_rate": 5,
            "run_time": "10m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test scenario - 50 users, rapid requests"""
        return {
            "users": 50,
            "spawn_rate": 10,
            "run_time": "15m"
        }

# Usage instructions
if __name__ == "__main__":
    print("Locust Load Testing Script for ML Pipeline API")
    print("=" * 50)
    print("\nTo run the load test:")
    print("1. Start your FastAPI server: python src/api.py")
    print("2. Run Locust: locust -f locust_load_test.py --host=http://localhost:8000")
    print("3. Open browser to http://localhost:8089")
    print("4. Configure test parameters and start the test")
    print("\nPredefined scenarios:")
    print("- Light load: 5 users, 2 minutes")
    print("- Medium load: 15 users, 5 minutes") 
    print("- Heavy load: 30 users, 10 minutes")
    print("- Stress test: 50 users, 15 minutes")
    print("\nThe test will simulate:")
    print("- Health check requests")
    print("- Single predictions")
    print("- Batch predictions")
    print("- Model information requests")
    print("- Prediction history requests")
    print("- Data visualization requests")
    print("- Retraining triggers") 