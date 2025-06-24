import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import io
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Cyber Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .feature-table {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_file):
    """Load a machine learning model from file"""
    try:
        # Try joblib first
        if model_file.name.endswith('.joblib'):
            model = joblib.load(model_file)
        elif model_file.name.endswith('.pkl'):
            model = pickle.load(model_file)
        else:
            st.error("Unsupported model format. Please use .pkl or .joblib files.")
            return None
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_csv_data(csv_file):
    """Load CSV data with caching"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics for model evaluation"""
    metrics = {}
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
    
    # ROC AUC (handle binary and multiclass)
    try:
        if len(np.unique(y_true)) == 2:
            if y_prob is not None and len(y_prob.shape) > 1:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_pred)
        else:
            if y_prob is not None:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                metrics['ROC AUC'] = "N/A (multiclass without probabilities)"
    except Exception:
        metrics['ROC AUC'] = "N/A"
    
    return metrics

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create a confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛡️ AI Cyber Intrusion Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["🔍 Single Sample Prediction", "📊 Model Accuracy Checker"])
    
    if page == "🔍 Single Sample Prediction":
        single_sample_prediction()
    else:
        model_accuracy_checker()

def single_sample_prediction():
    """Page 1: Single Sample Prediction"""
    st.markdown('<h2 class="sub-header">🔍 Single Sample Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Data")
        csv_file = st.file_uploader("Upload CSV file with 77 features", type=['csv'], key="single_csv")
        
        if csv_file is not None:
            df = load_csv_data(csv_file)
            if df is not None:
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Validate 77 features
                if df.shape[1] < 77:
                    st.warning(f"⚠️ Expected 77 features, but found {df.shape[1]}. Please check your data.")
                
                # Row index selection
                max_rows = len(df) - 1
                row_index = st.number_input(
                    f"Select row index (0 to {max_rows})", 
                    min_value=0, 
                    max_value=max_rows, 
                    value=0,
                    key="row_index"
                )
                
                # Display selected row features
                if st.button("🔍 Show Features", key="show_features"):
                    st.subheader(f"📋 Features for Row {row_index}")
                    
                    # Get the selected row
                    selected_row = df.iloc[row_index]
                    
                    # Create a nice display of features
                    feature_df = pd.DataFrame({
                        'Feature Index': range(len(selected_row)),
                        'Feature Name': selected_row.index,
                        'Value': selected_row.values
                    })
                    
                    with st.expander("📊 Feature Values (Click to expand)", expanded=True):
                        st.dataframe(
                            feature_df,
                            use_container_width=True,
                            height=400
                        )
    
    with col2:
        st.subheader("🤖 Upload Model")
        model_file = st.file_uploader("Upload trained model (.pkl or .joblib)", 
                                    type=['pkl', 'joblib'], 
                                    key="single_model")
        
        if model_file is not None and csv_file is not None:
            model = load_model(model_file)
            df = load_csv_data(csv_file)
            
            if model is not None and df is not None:
                st.success("✅ Model loaded successfully!")
                
                if st.button("🚀 Make Prediction", key="predict_single"):
                    try:
                        # Get the selected row
                        row_index = st.session_state.get("row_index", 0)
                        selected_row = df.iloc[row_index:row_index+1]
                        
                        # Make prediction
                        prediction = model.predict(selected_row)[0]
                        
                        # Try to get prediction probability
                        try:
                            prediction_proba = model.predict_proba(selected_row)[0]
                            max_proba = np.max(prediction_proba)
                        except:
                            prediction_proba = None
                            max_proba = None
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        # Create metrics display
                        col_pred1, col_pred2 = st.columns(2)
                        
                        with col_pred1:
                            st.metric("Prediction", str(prediction))
                        
                        with col_pred2:
                            if max_proba is not None:
                                st.metric("Confidence", f"{max_proba:.2%}")
                        
                        # Detailed results table
                        result_df = pd.DataFrame({
                            'Model Name': [model_file.name],
                            'Prediction': [prediction],
                            'Confidence': [f"{max_proba:.2%}" if max_proba else "N/A"]
                        })
                        
                        st.table(result_df)
                        
                        # Show probability distribution if available
                        if prediction_proba is not None:
                            with st.expander("📊 Prediction Probabilities"):
                                prob_df = pd.DataFrame({
                                    'Class': [f"Class {i}" for i in range(len(prediction_proba))],
                                    'Probability': prediction_proba
                                })
                                st.bar_chart(prob_df.set_index('Class'))
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")

def model_accuracy_checker():
    """Page 2: Model Accuracy Checker"""
    st.markdown('<h2 class="sub-header">📊 Model Accuracy Checker</h2>', unsafe_allow_html=True)
    
    # File uploads
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Test Data")
        test_csv = st.file_uploader("Upload test CSV with ground-truth labels", 
                                  type=['csv'], 
                                  key="test_csv")
        
        if test_csv is not None:
            test_df = load_csv_data(test_csv)
            if test_df is not None:
                st.success(f"✅ Test data loaded! Shape: {test_df.shape}")
                
                # Label column selection
                label_column = st.selectbox(
                    "Select the ground-truth label column",
                    options=test_df.columns.tolist(),
                    key="label_column"
                )
    
    with col2:
        st.subheader("🤖 Upload Models")
        model_files = st.file_uploader("Upload trained models (.pkl or .joblib)", 
                                     type=['pkl', 'joblib'], 
                                     accept_multiple_files=True,
                                     key="multiple_models")
        
        if model_files:
            st.success(f"✅ {len(model_files)} model(s) uploaded!")
    
    # Model evaluation
    if test_csv is not None and model_files and len(model_files) > 0:
        test_df = load_csv_data(test_csv)
        
        if test_df is not None and st.button("🚀 Evaluate Models", key="evaluate_models"):
            label_column = st.session_state.get("label_column")
            
            if label_column not in test_df.columns:
                st.error("❌ Selected label column not found in the dataset!")
                return
            
            # Prepare data
            y_true = test_df[label_column]
            X_test = test_df.drop(columns=[label_column])
            
            # Results storage
            results = []
            confusion_matrices = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Evaluate each model
            for i, model_file in enumerate(model_files):
                status_text.text(f"Evaluating {model_file.name}...")
                
                model = load_model(model_file)
                if model is not None:
                    try:
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Try to get probabilities
                        try:
                            y_prob = model.predict_proba(X_test)
                        except:
                            y_prob = None
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_true, y_pred, y_prob)
                        
                        # Store results
                        result = {
                            'Model Name': model_file.name,
                            **metrics
                        }
                        results.append(result)
                        
                        # Store confusion matrix data
                        confusion_matrices[model_file.name] = (y_true, y_pred)
                        
                    except Exception as e:
                        st.error(f"❌ Error evaluating {model_file.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(model_files))
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if results:
                st.subheader("📈 Model Performance Summary")
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display metrics table
                st.dataframe(results_df, use_container_width=True)
                
                # Display individual model details
                st.subheader("🔍 Detailed Model Analysis")
                
                for result in results:
                    model_name = result['Model Name']
                    
                    with st.expander(f"📊 {model_name} - Detailed Metrics"):
                        # Metrics in columns
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Accuracy", f"{result['Accuracy']:.4f}")
                        with col2:
                            st.metric("F1 Score", f"{result['F1 Score']:.4f}")
                        with col3:
                            st.metric("Precision", f"{result['Precision']:.4f}")
                        with col4:
                            st.metric("Recall", f"{result['Recall']:.4f}")
                        with col5:
                            roc_auc = result['ROC AUC']
                            if isinstance(roc_auc, str):
                                st.metric("ROC AUC", roc_auc)
                            else:
                                st.metric("ROC AUC", f"{roc_auc:.4f}")
                        
                        # Confusion Matrix
                        if model_name in confusion_matrices:
                            st.subheader("🎯 Confusion Matrix")
                            y_true_cm, y_pred_cm = confusion_matrices[model_name]
                            
                            fig = create_confusion_matrix_plot(y_true_cm, y_pred_cm, model_name)
                            st.pyplot(fig)
                            plt.close(fig)  # Prevent memory leaks
                
                # Best model highlight
                if len(results) > 1:
                    st.subheader("🏆 Best Performing Model")
                    best_model = max(results, key=lambda x: x['Accuracy'])
                    
                    st.success(f"🥇 **Best Model:** {best_model['Model Name']} with {best_model['Accuracy']:.4f} accuracy")

if __name__ == "__main__":
    main()
