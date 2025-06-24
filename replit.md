# AI Cyber Intrusion Detection System

## Overview

This is a Streamlit-based web application for AI-powered cyber intrusion detection. The system provides a user-friendly interface for analyzing network security data, visualizing threats, and performing machine learning-based intrusion detection. The application is designed to help security analysts identify and analyze potential cyber threats through interactive data visualization and model predictions.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for its simplicity in creating data science web applications
- **Styling**: Custom CSS embedded within the Streamlit app for enhanced UI/UX
- **Layout**: Wide layout with expandable sidebar for better data visualization space
- **Components**: Interactive widgets for data upload, model selection, and results display

### Backend Architecture
- **Runtime**: Python 3.11 with scientific computing stack
- **Data Processing**: Pandas and NumPy for data manipulation and numerical operations
- **Machine Learning**: Scikit-learn for model training, evaluation, and predictions
- **Model Persistence**: Joblib and Pickle for saving/loading trained models
- **Visualization**: Matplotlib and Seaborn for generating charts and plots

### Technology Stack
- Python 3.11 as the primary runtime
- Streamlit for web application framework
- Scientific Python stack (NumPy, Pandas, Scikit-learn)
- Data visualization libraries (Matplotlib, Seaborn)
- Model serialization tools (Joblib, Pickle)

## Key Components

### 1. Data Processing Module
- Handles CSV data upload and preprocessing
- Implements data cleaning and feature engineering
- Supports label encoding for categorical variables
- Provides data validation and error handling

### 2. Machine Learning Pipeline
- Model training and evaluation capabilities
- Support for multiple ML algorithms
- Performance metrics calculation (accuracy, F1-score, precision, recall, ROC-AUC)
- Confusion matrix generation and visualization

### 3. Visualization Engine
- Interactive data exploration tools
- Statistical plots and charts
- Model performance visualizations
- Feature importance analysis

### 4. User Interface Components
- File upload interface
- Model configuration panels
- Results dashboard with metrics cards
- Interactive data tables with scrolling

## Data Flow

1. **Data Input**: Users upload CSV files containing network traffic or security logs
2. **Data Preprocessing**: Automatic data cleaning, encoding, and feature extraction
3. **Model Training**: ML models are trained on the processed data
4. **Prediction**: Trained models classify new data points as normal or intrusive
5. **Visualization**: Results are displayed through interactive charts and metrics
6. **Export**: Model performance metrics and predictions can be analyzed and exported

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for data science applications
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing library
- **Scikit-learn**: Machine learning library with algorithms and evaluation tools
- **Matplotlib**: Plotting library for static visualizations
- **Seaborn**: Statistical data visualization library
- **Joblib**: Efficient serialization for NumPy arrays and scikit-learn models

### System Dependencies (via Nix)
- Cairo, FFmpeg, FreeType, Ghostscript for graphics and media processing
- GTK3 and GObject Introspection for GUI components
- Tcl/Tk for additional UI capabilities
- Development tools (pkg-config, qhull) for building dependencies

## Deployment Strategy

### Platform
- **Target**: Replit autoscale deployment
- **Port**: Application runs on port 5000
- **Server Configuration**: Headless mode with external access (0.0.0.0)

### Environment
- **Package Management**: UV lock file ensures reproducible dependencies
- **Runtime**: Python 3.11 with Nix-managed system dependencies
- **Configuration**: Streamlit config optimized for production deployment

### Workflow
- **Development**: Interactive development with hot-reload via Streamlit
- **Deployment**: Automated deployment through Replit workflows
- **Scaling**: Autoscale deployment target for handling variable traffic

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 24, 2025. Initial setup