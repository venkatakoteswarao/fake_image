#!/usr/bin/env python3
"""
Web-based DeepFake Image Detection
Upload images and get predictions from all trained models
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add Pretrained_Models to path
sys.path.append('Pretrained_Models')

from config import *

# Set page config
st.set_page_config(
    page_title="DeepFake Image Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    model_files = {
        "VGG16": "vgg16_final_model.h5",
        "VGG19": "vgg19_final_model.h5",
        "ResNet50": "resnet50_final_model.h5",
        "InceptionV3": "inceptionv3_final_model.h5",
        "Custom Model": "custom_final_model.h5"
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                models[model_name] = tf.keras.models.load_model(model_path)
            except Exception as e:
                st.error(f"❌ Failed to load {model_name}: {e}")
        else:
            st.warning(f"⚠️ Model not found: {model_path}")
    
    return models

def get_model_info():
    """Get model information including actual trained accuracy"""
    return {
        "VGG16": {"accuracy": 85.56, "description": "Deep CNN with 16 layers"},
        "VGG19": {"accuracy": 85.58, "description": "Deep CNN with 19 layers"},
        "ResNet50": {"accuracy": 50.00, "description": "Residual network with 50 layers"},
        "InceptionV3": {"accuracy": 77.39, "description": "Inception architecture for efficient recognition"},
        "Custom Model": {"accuracy": 95.02, "description": "Custom CNN with 9 layers"}
    }

def preprocess_image(image, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_model(model, model_name, image_array):
    """Make prediction with a single model"""
    try:
        prediction = model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])
        
        is_fake = confidence > 0.5
        result = "FAKE" if is_fake else "REAL"
        
        return {
            'model': model_name,
            'prediction': result,
            'confidence': confidence,
            'fake_probability': confidence,
            'real_probability': 1 - confidence
        }
    except Exception as e:
        st.error(f"Error with {model_name}: {e}")
        return None

def ensemble_prediction(results):
    """Create ensemble prediction"""
    if not results:
        return None
    
    # Weights based on expected performance
    weights = {
        'VGG16': 0.25,
        'VGG19': 0.25,
        'ResNet50': 0.20,
        'InceptionV3': 0.15,
        'Custom Model': 0.15
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for result in results:
        if result:
            weight = weights.get(result['model'], 0.2)
            weighted_sum += result['fake_probability'] * weight
            total_weight += weight
    
    if total_weight == 0:
        return None
    
    ensemble_confidence = weighted_sum / total_weight
    is_fake = ensemble_confidence > 0.5
    result = "FAKE" if is_fake else "REAL"
    
    return {
        'prediction': result,
        'confidence': ensemble_confidence,
        'fake_probability': ensemble_confidence,
        'real_probability': 1 - ensemble_confidence
    }

def main():
    # Header
    st.title("🔍 DeepFake Image Detection")
    st.markdown("Upload an image and select a model to detect if it's real or fake")
    
    # Set TensorFlow to use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if not models:
        st.error("❌ No models available. Please ensure models are trained and saved.")
        return
    
    st.success(f"✅ Loaded {len(models)} models successfully!")
    
    # Get model information
    model_info = get_model_info()
    
    # Sidebar - Model Selection
    st.sidebar.header("🤖 Select Model")
    
    # Model selection
    available_models = list(models.keys())
    selected_model = st.sidebar.selectbox(
        "Choose a model to use:",
        available_models,
        help="Select which AI model to use for prediction"
    )
    
    # Display selected model info
    if selected_model in model_info:
        info = model_info[selected_model]
        st.sidebar.subheader("📊 Model Information")
        st.sidebar.metric("Model Accuracy", f"{info['accuracy']:.2f}%")
        st.sidebar.write(f"**Description:** {info['description']}")
        
        # Model file size
        model_path = f"saved_models/{selected_model.lower().replace(' ', '_')}_final_model.h5"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            st.sidebar.metric("Model Size", f"{size_mb:.1f} MB")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image to detect if it's real or fake"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            image_array = preprocess_image(image)
            
            # Predict button
            if st.button(f"🔍 Analyze with {selected_model}", type="primary"):
                with st.spinner(f"Analyzing image with {selected_model}..."):
                    # Make prediction with selected model
                    result = predict_with_model(models[selected_model], selected_model, image_array)
                    
                    if result:
                        # Store result in session state
                        st.session_state['selected_model'] = selected_model
                        st.session_state['result'] = result
                        st.session_state['image_array'] = image_array
    
    with col2:
        st.header("📊 Results")
        
        if 'result' in st.session_state and 'selected_model' in st.session_state:
            result = st.session_state['result']
            model_name = st.session_state['selected_model']
            
            # Display selected model info
            st.subheader(f"🤖 {model_name} Prediction")
            
            if model_name in model_info:
                info = model_info[model_name]
                col_acc, col_desc = st.columns([1, 1])
                with col_acc:
                    st.metric("Model Accuracy", f"{info['accuracy']:.2f}%")
                with col_desc:
                    st.write(f"**{info['description']}**")
            
            # Prediction result
            st.subheader("🎯 Prediction Result")
            
            if result['prediction'] == 'FAKE':
                st.error(f"**Result: {result['prediction']}**")
                st.error("⚠️ This image appears to be a deepfake!")
            else:
                st.success(f"**Result: {result['prediction']}**")
                st.success("✅ This image appears to be real!")
            
            # Confidence metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']:.3f}")
            with col2:
                st.metric("Fake Probability", f"{result['fake_probability']:.3f}")
            with col3:
                st.metric("Real Probability", f"{result['real_probability']:.3f}")
            
            # Confidence bar
            st.subheader("📊 Confidence Visualization")
            fake_prob = result['fake_probability']
            real_prob = result['real_probability']
            
            col_fake, col_real = st.columns(2)
            with col_fake:
                st.progress(fake_prob)
                st.write(f"Fake: {fake_prob:.1%}")
            with col_real:
                st.progress(real_prob)
                st.write(f"Real: {real_prob:.1%}")
            
            # Additional analysis
            st.subheader("🔍 Analysis Details")
            if result['confidence'] > 0.8:
                st.success("High confidence prediction")
            elif result['confidence'] > 0.6:
                st.warning("Medium confidence prediction")
            else:
                st.info("Low confidence prediction - result may be uncertain")
        
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Footer with all model accuracies
    st.markdown("---")
    st.subheader("📋 All Available Models")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    models_data = [
        ("VGG16", 85.56),
        ("VGG19", 85.58),
        ("ResNet50", 50.00),
        ("InceptionV3", 77.39),
        ("Custom", 95.02)
    ]
    
    for i, (name, acc) in enumerate(models_data):
        with [col1, col2, col3, col4, col5][i]:
            st.metric(name, f"{acc:.2f}%")

if __name__ == "__main__":
    main()