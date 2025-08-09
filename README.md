# 🏨 Hotel Reservation Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://docker.com)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-red.svg)](https://jenkins.io)

A complete end-to-end machine learning project that predicts hotel reservation cancellations using advanced ML algorithms and production-ready deployment infrastructure.

## 🎯 Project Overview

This project implements a sophisticated machine learning system to predict whether a hotel guest will cancel their reservation. The system helps hotels optimize their booking strategies, reduce revenue loss, and improve operational efficiency.

### Key Features

- **🔮 Predictive Analytics**: LightGBM-based model with hyperparameter optimization
- **🌐 Web Interface**: Interactive Flask web application for real-time predictions
- **📊 MLOps Pipeline**: Complete MLflow integration for experiment tracking
- **🚀 Production Ready**: Docker containerization and Jenkins CI/CD pipeline
- **📈 Data Processing**: Advanced preprocessing with SMOTE for handling class imbalance
- **🔧 Modular Design**: Clean, maintainable codebase with proper logging and error handling

## 📁 Project Structure

```
Hotel Reservation Prediction/
├── 🚀 application.py              # Flask web application
├── 📋 main.py                     # Project entry point
├── ⚙️ config/                     # Configuration management
│   ├── config.yaml                # Main configuration file
│   └── training_config.py         # ML model parameters
├── 🔄 pipeline/                   # ML pipeline orchestration
│   └── training_pipeline.py       # End-to-end training workflow
├── 📊 src/                        # Core source code
│   ├── data_ingestion.py          # Data download and splitting
│   ├── data_processing.py         # Feature engineering & preprocessing
│   ├── training_data.py           # Model training and evaluation
│   ├── logger.py                  # Logging configuration
│   └── custom_exception.py        # Custom error handling
├── 🛠️ utils/                      # Utility functions
│   └── common_functions.py        # Shared helper functions
├── 📓 notebook/                   # Jupyter notebooks
│   ├── analysis.ipynb             # Exploratory Data Analysis
│   ├── imblearn.ipynb             # Imbalanced learning experiments
│   └── logging.ipynb              # Logging system testing
├── 🎨 templates/                  # Web UI templates
│   └── index.html                 # Main prediction interface
├── 🎨 static/                     # CSS and static assets
│   └── style.css                  # Web application styling
├── 📦 artifacts/                  # Generated model artifacts
├── 📈 mlruns/                     # MLflow experiment tracking
├── 🐳 Dockerfile                  # Container configuration
├── 🔧 Jenkinsfile                 # CI/CD pipeline
└── 📋 requirements.txt            # Python dependencies
```

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **Python 3.11+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: ML algorithms and preprocessing
- **LightGBM**: Gradient boosting framework for predictions
- **Imbalanced-learn**: Handling class imbalance with SMOTE
- **MLflow**: Experiment tracking and model management

### Web Development
- **Flask**: Lightweight web framework
- **HTML/CSS**: Frontend user interface
- **Jinja2**: Template engine for dynamic content

### DevOps & Deployment
- **Docker**: Application containerization
- **Jenkins**: Continuous Integration/Continuous Deployment
- **Git**: Version control system

### Development Tools
- **Jupyter Notebook**: Interactive development and analysis
- **PyYAML**: Configuration management
- **Python-box**: Enhanced configuration handling

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/OmarAymanHassan/Hotel-Reservation-Prediction.git
   cd Hotel-Reservation-Prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. **Train the model**
   ```bash
   python pipeline/training_pipeline.py
   ```

5. **Run the web application**
   ```bash
   python application.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t hotel-reservation-prediction .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 hotel-reservation-prediction
   ```

## 🎯 How It Works

### Data Pipeline

1. **Data Ingestion** (`src/data_ingestion.py`)
   - Downloads hotel reservation data from Google Drive
   - Splits data into training and testing sets (80/20 ratio)
   - Handles data validation and error management

2. **Data Processing** (`src/data_processing.py`)
   - Removes irrelevant columns (Booking_ID)
   - Handles categorical encoding (Ordinal Encoding, One-Hot Encoding)
   - Applies feature scaling and normalization
   - Implements SMOTE for handling class imbalance
   - Manages skewness detection and transformation

3. **Model Training** (`src/training_data.py`)
   - Utilizes LightGBM classifier with hyperparameter optimization
   - Implements RandomizedSearchCV for parameter tuning
   - Tracks experiments using MLflow
   - Evaluates model performance with comprehensive metrics

### Prediction Features

The model uses 17 input features to predict reservation cancellations:

- **Booking Details**: Lead time, arrival date/month/year
- **Guest Information**: Number of adults, children, repeated guest status
- **Stay Details**: Weekend nights, weekday nights, room type
- **Services**: Meal plan type, special requests, parking requirements
- **History**: Previous cancellations, previous bookings
- **Pricing**: Average price per room
- **Market**: Market segment type

### Web Interface

The Flask application provides an intuitive web interface where users can:
- Input reservation details through a user-friendly form
- Get real-time predictions on cancellation likelihood
- View results with clear interpretation

## 📊 Model Performance

The project implements advanced ML techniques:

- **Algorithm**: LightGBM (Light Gradient Boosting Machine)
- **Hyperparameter Optimization**: RandomizedSearchCV with cross-validation
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Experiment Tracking**: MLflow for comprehensive model management

## 🔧 Configuration

The project uses YAML-based configuration management:

### `config/config.yaml`
```yaml
data_ingestion:
  root_dir: artifacts/raw
  train_ratio: 0.8
  source_url: [Google Drive Link]

data_processing:
  processed_dir: artifacts/processing
  target_col: booking_status
  skewness_threshold: 10

data_training:
  root_dir: artifacts/training
  random_state: 42
```

### `config/training_config.py`
- LightGBM hyperparameters configuration
- RandomizedSearchCV parameters
- Cross-validation settings

## 🧪 Notebooks & Analysis

### `notebook/analysis.ipynb`
- Comprehensive Exploratory Data Analysis (EDA)
- Feature distribution analysis
- Correlation studies
- Data quality assessment

### `notebook/imblearn.ipynb`
- Class imbalance analysis
- SMOTE implementation experiments
- Performance comparison studies

### `notebook/logging.ipynb`
- Logging system testing
- Error handling validation

## 🚀 CI/CD Pipeline

### Jenkins Pipeline (`Jenkinsfile`)

The project includes a complete CI/CD pipeline:

1. **Source Code Management**: Automatic GitHub repository cloning
2. **Environment Setup**: Virtual environment creation and dependency installation
3. **Testing**: Automated testing (extensible for unit tests)
4. **Deployment**: Ready for production deployment integration

### Docker Configuration

- **Multi-stage build** for optimized image size
- **Automatic model training** during image build
- **Production-ready configuration** with proper environment variables
- **LightGBM dependencies** properly configured

## 📈 MLflow Integration

The project leverages MLflow for comprehensive experiment management:

- **Experiment Tracking**: All training runs logged with parameters and metrics
- **Model Registry**: Centralized model versioning and management
- **Artifact Storage**: Model artifacts and preprocessing pipelines stored
- **Comparison Tools**: Easy comparison between different model iterations

Access MLflow UI:
```bash
mlflow ui
```

## 🔍 Logging & Monitoring

Comprehensive logging system implemented throughout the project:

- **Structured Logging**: Consistent log format across all modules
- **Error Tracking**: Detailed error messages with stack traces
- **Custom Exceptions**: Specific error handling for different failure scenarios
- **Log Levels**: Appropriate logging levels (INFO, ERROR, DEBUG)

## 📝 Usage Examples

### Making Predictions via Web Interface

1. Navigate to `http://localhost:5000`
2. Fill in the reservation details form:
   - Lead time (days in advance)
   - Number of adults and children
   - Arrival date/month/year
   - Room type and meal plan
   - Special requests and parking needs
   - Guest history information
3. Click "Predict" to get cancellation probability

