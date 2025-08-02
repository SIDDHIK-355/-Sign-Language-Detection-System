import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")

# Load model and classes
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/sign_language_model.h5')
    with open('models/classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    return model, classes

def check_operational_time():
    """Check if current time is between 6 PM and 10 PM"""
    current_time = datetime.now().time()
    start_time = datetime.strptime("18:00", "%H:%M").time()
    end_time = datetime.strptime("22:00", "%H:%M").time()
    return start_time <= current_time <= end_time

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Main app
st.title("🤟 Sign Language Detection System")
st.markdown("---")

# Check operational time
is_operational = check_operational_time()
current_time = datetime.now().strftime("%I:%M %p")

col1, col2 = st.columns([3, 1])
with col1:
    st.info(f"Current Time: {current_time}")
with col2:
    if is_operational:
        st.success("✅ System Active")
    else:
        st.error("❌ System Inactive")

if not is_operational:
    st.warning("⏰ System operates between 6 PM - 10 PM only!")
    st.stop()

# Load model
try:
    model, classes = load_model()
except:
    st.error("Please train the model first by running model_training.ipynb")
    st.stop()

# Sidebar
st.sidebar.title("Options")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Real-time Camera"])

if mode == "Upload Image":
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict
        processed_image = preprocess_image(image_np)
        prediction = model.predict(processed_image)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        with col2:
            st.subheader("Prediction Results")
            st.metric("Predicted Letter", predicted_class)
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show top 3 predictions
            st.subheader("Top 3 Predictions")
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            for idx in top_3_idx:
                st.write(f"{classes[idx]}: {prediction[0][idx]*100:.2f}%")

else:  # Real-time Camera
    st.header("📹 Real-time Camera")
    
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a copy for prediction
            pred_frame = frame.copy()
            
            # Preprocess and predict
            processed = preprocess_image(pred_frame)
            prediction = model.predict(processed)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Add text to frame
            cv2.putText(frame_rgb, f"Letter: {predicted_class} ({confidence:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            FRAME_WINDOW.image(frame_rgb)
        
        cap.release()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Sign Language Recognition")