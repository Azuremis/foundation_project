"""
Main Streamlit application for MNIST digit classification.
"""
import os
import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas
from dotenv import load_dotenv

# Import custom modules
from inference import load_model, predict_from_upload, predict_from_canvas, update_prediction_with_user_label
from db.db_utils import get_prediction_stats, get_prediction_history
from data_utils import image_to_bytes

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier", 
    page_icon="✏️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4b8bf4;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
    }
    .success-box {
        background-color: #d1f0d1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1e7f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<p class="main-header">MNIST Digit Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Draw or upload a handwritten digit for classification</p>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_digit' not in st.session_state:
    st.session_state.predicted_digit = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None

# Sidebar content
with st.sidebar:
    st.header("Model Information")
    st.info("This app uses a CNN to classify handwritten digits.")
    
    # Try to load model stats
    try:
        stats = get_prediction_stats()
        st.write("### Prediction Statistics")
        st.write(f"Total predictions: {stats['total']}")
        st.write(f"Correct predictions: {stats['correct']}")
        if stats['total'] > 0:
            st.write(f"Accuracy: {stats['accuracy']*100:.2f}%")
    except Exception as e:
        st.warning(f"Could not load prediction stats: {e}")

# Cache model loading to improve performance
@st.cache_resource
def get_model():
    try:
        model = load_model()
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = get_model()

# Create two columns for the input methods
col1, col2 = st.columns(2)

with col1:
    st.header("Draw a digit")
    
    # Set up canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    # Predict button for canvas
    if st.button("Predict from Drawing"):
        if model is not None and canvas_result.image_data is not None:
            # Make prediction
            digit, confidence = predict_from_canvas(model, canvas_result.image_data)
            
            # Save results in session state
            st.session_state.prediction_made = True
            st.session_state.predicted_digit = digit
            st.session_state.confidence = confidence
            st.session_state.image_data = image_to_bytes(Image.fromarray(canvas_result.image_data))

with col2:
    st.header("Upload an image")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image of a handwritten digit", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=280)
        
        # Predict button for uploaded image
        if st.button("Predict from Upload"):
            if model is not None:
                # Make prediction
                digit, confidence = predict_from_upload(model, uploaded_file)
                
                # Save results in session state
                st.session_state.prediction_made = True
                st.session_state.predicted_digit = digit
                st.session_state.confidence = confidence
                
                # Read uploaded file into bytes for database storage
                uploaded_file.seek(0)
                st.session_state.image_data = uploaded_file.read()

# Display prediction results if a prediction has been made
if st.session_state.prediction_made:
    st.markdown("---")
    st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown(
            f"""
            <div class="success-box">
                <h3>Predicted Digit: {st.session_state.predicted_digit}</h3>
                <p>Confidence: {st.session_state.confidence * 100:.2f}%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with result_col2:
        # Get user feedback
        st.markdown('<p>Is this prediction correct?</p>', unsafe_allow_html=True)
        user_label = st.number_input("Enter the correct digit (0-9)", min_value=0, max_value=9, value=st.session_state.predicted_digit)
        
        if st.button("Submit Feedback"):
            # Update database with user feedback
            success = update_prediction_with_user_label(
                predicted_digit=st.session_state.predicted_digit,
                user_label=user_label,
                confidence=st.session_state.confidence,
                image_data=st.session_state.image_data
            )
            
            if success:
                st.success("Thank you for your feedback!")
            else:
                st.error("Failed to save feedback. Please try again.")
    
    # Reset button
    if st.button("Make Another Prediction"):
        st.session_state.prediction_made = False
        st.session_state.predicted_digit = None
        st.session_state.confidence = None
        st.session_state.image_data = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, PyTorch, and PostgreSQL")

# Add prediction history section
st.markdown("---")
st.markdown('<p class="sub-header">Prediction History</p>', unsafe_allow_html=True)

# Fetch prediction history from the database
history = get_prediction_history(limit=20)

if history:
    # Create a dataframe for the history
    history_data = {
        "Timestamp": [h["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for h in history],
        "Predicted Digit": [h["predicted_digit"] for h in history],
        "True Label": [h["user_label"] for h in history],
        "Confidence": [f"{h['confidence']*100:.2f}%" if h["confidence"] is not None else "N/A" for h in history],
        "Correct": ["✅" if h["correct"] else "❌" for h in history]
    }
    
    # Display history as a table
    st.dataframe(
        history_data,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Timestamp"),
            "Predicted Digit": st.column_config.NumberColumn("Predicted Digit"),
            "True Label": st.column_config.NumberColumn("True Label"),
            "Confidence": st.column_config.TextColumn("Confidence"),
            "Correct": st.column_config.TextColumn("Correct")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No prediction history found. Make some predictions and provide feedback to build history.") 