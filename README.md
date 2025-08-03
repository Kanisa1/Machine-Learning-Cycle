# 🫁 Chest X-Ray Pneumonia Detection - Complete ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle)

**🎯 Advanced Machine Learning Pipeline for Medical Image Classification**

[🎥 **Video Demo**](https://youtu.be/8DVwQEWzCCQ) • [📚 **API Docs**](http://localhost:8000/docs#/) • [📊 **Dataset**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) • [🚀 **Live Dashboard**](file:///C:/Users/HP/Machine-Learning-Cycle/src/dashboard.html)

</div>

---

## 📋 **Project Overview**

A complete end-to-end machine learning pipeline for automated pneumonia detection from chest X-ray images. This production-ready system includes real-time predictions, model retraining capabilities, comprehensive monitoring, and multi-cloud deployment support.

### 🎯 **Key Features**
- ✅ **Advanced ML Models** with Logistic Regression, Random Forest, and SVM support
- ✅ **Real-time Image Analysis** via FastAPI backend
- ✅ **Interactive Web Dashboard** with live monitoring
- ✅ **Automated Retraining Pipeline** with performance tracking
- ✅ **Production-Ready Deployment** with Docker and Kubernetes support
- ✅ **Comprehensive Load Testing** with Locust framework
- ✅ **Image Preprocessing** with OpenCV and advanced feature extraction

### 📊 **Performance Metrics**
- **Model Types**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Dataset Size**: 5,856 chest X-ray images (train/test split)
- **API Endpoints**: Health check, prediction, batch processing, retraining
- **Load Testing**: Locust framework with concurrent user simulation
- **Deployment**: Docker containers with nginx reverse proxy

---

## 🖼️ **Sample Results**

### **Dashboard Interface**
![Dashboard Screenshot](docs/dashboard_screenshot.png)

### **Prediction Examples**

| **Normal Chest X-Ray** | **Pneumonia Detection** | **Analysis Results** |
|:---:|:---:|:---:|
| ![Normal](docs/normal_chest_xray.jpg) | ![Pneumonia](docs/pneumonia_chest_xray.jpg) | ![Results](docs/sample_analysis.jpg) |
| Normal X-Ray Image | Pneumonia X-Ray Image | Real-time Analysis |

### **Feature Visualizations**
- **🔬 Image Preprocessing**: Advanced OpenCV-based image processing pipeline
- **📈 Feature Extraction**: Comprehensive feature engineering for medical images  
- **🗺️ Model Comparison**: Multiple ML algorithms with performance benchmarking

---

## 🚀 **Quick Start Guide**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/Kanisa1/Machine-Learning-Cycle.git
cd Machine-Learning-Cycle

# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend
cd src
python api.py

# API will be available at: http://localhost:8000
# Interactive API docs: http://localhost:8000/docs#/
```

### **2. Using the Dashboard**
```bash
# Open the web dashboard
start src/dashboard.html

# Or navigate to: file:///C:/Users/HP/Machine-Learning-Cycle/src/dashboard.html
```

### **3. API Endpoints**
- **Health Check**: `GET /health`
- **Model Info**: `GET /model_info`
- **Single Prediction**: `POST /predict`
- **Image Prediction**: `POST /predict_image`
- **Batch Prediction**: `POST /predict_batch`
- **Retraining**: `POST /trigger_retraining`
- **Prediction History**: `GET /prediction_history`

---

## 🏗️ **Project Structure**

```
Machine-Learning-Cycle/
├── src/                    # Main application code
│   ├── api.py             # FastAPI application
│   ├── model.py           # Model management
│   ├── prediction.py      # Prediction service
│   ├── preprocessing.py   # Image preprocessing
│   └── dashboard.html     # Web dashboard
├── data/                  # Dataset and processed data
│   ├── chest_xray/        # Original dataset
│   ├── processed/         # Preprocessed data
│   └── retraining_uploads/ # Uploaded images
├── models/                # Trained model files
├── deployment/            # Deployment configurations
├── monitoring/            # Monitoring and logging
├── notebook/              # Jupyter notebooks
├── docs/                  # Documentation and images
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-container setup
└── locust_load_test.py   # Load testing script
```

---

## 🔧 **Technical Stack**

### **Backend**
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **OpenCV**: Computer vision library for image processing
- **Scikit-learn**: Machine learning library for model training

### **Frontend**
- **HTML/CSS/JavaScript**: Interactive web dashboard
- **Chart.js**: Data visualization and monitoring charts

### **DevOps**
- **Docker**: Containerization for consistent deployment
- **Nginx**: Reverse proxy and load balancing
- **Locust**: Load testing framework

### **Data Processing**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Pillow**: Image processing library

---

## 📈 **Model Performance**

The system supports multiple machine learning algorithms:

- **Logistic Regression**: Fast and interpretable baseline model
- **Random Forest**: Robust ensemble method with feature importance
- **Gradient Boosting**: High-performance boosting algorithm
- **Support Vector Machine**: Advanced classification with kernel methods

Each model can be trained, evaluated, and compared through the API endpoints.

---

## 🚀 **Deployment**

### **Docker Deployment**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual containers
docker build -t chest-xray-api .
docker run -p 8000:8000 chest-xray-api
```

### **Load Testing**
```bash
# Run load tests with Locust
locust -f locust_load_test.py --host=http://localhost:8000
```

---

## 📚 **API Documentation**

Visit [http://localhost:8000/docs#/](http://localhost:8000/docs#/) for interactive API documentation with:
- All available endpoints
- Request/response schemas
- Try-it-out functionality
- Authentication details

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 **Acknowledgments**

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- **FastAPI**: Modern web framework for building APIs
- **Scikit-learn**: Machine learning library
- **OpenCV**: Computer vision library

---

<div align="center">

**Made with ❤️ for Medical AI Research**

[⬆ Back to Top](#-chest-x-ray-pneumonia-detection---complete-ml-pipeline)

</div>
