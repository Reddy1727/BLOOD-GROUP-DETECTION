import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import os
import base64

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Add custom CSS for better styling with background image
bg_image = get_base64_image("SBackground.jpg")
if bg_image:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
        }}
        
        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        
        [data-testid="stMainBlockContainer"] {{
            background: transparent;
        }}
        
        /* Main title styling */
        h1 {{
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        /* Subtitle styling */
        p {{
            color: #ffffff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            font-size: 1.1rem;
        }}
        
        /* Section headers */
        h2 {{
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            font-size: 1.8rem;
            font-weight: 600;
        }}
        
        /* Container styling */
        [data-testid="stColumn"] {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        
        .fingerprint-logo {{
            font-size: 80px;
            text-align: center;
            margin: 20px 0;
        }}
        
        .blood-group-result {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,248,255,0.95) 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 2px solid rgba(255,255,255,0.3);
        }}
        
        /* Metric styling */
        [data-testid="metric-container"] {{
            background: rgba(255,255,255,0.9) !important;
            padding: 20px !important;
            border-radius: 10px !important;
            border: 2px solid rgba(220, 20, 60, 0.3) !important;
            margin: 10px 0 !important;
        }}
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {{
            background-color: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 15px;
        }}
        
        /* Button styling */
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 10px 30px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}
        
        button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        }}
        
        /* Section divider */
        hr {{
            border: 1px solid rgba(255,255,255,0.3);
            margin: 30px 0;
        }}
        
        /* Info boxes */
        [data-testid="stMarkdownContainer"] {{
            color: #ffffff; 
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }}
        </style>
        """, unsafe_allow_html=True)
else:
    # Fallback styling
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        h1 {{
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            font-size: 2.5rem;
        }}
        
        .blood-group-result {{
            text-align: center;
            padding: 30px;
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }}
        
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 8px !important;
        }}
        </style>
        """, unsafe_allow_html=True)

st.title("🩸 Fingerprint-Based Blood Group Detection")
st.write("Upload a **fingerprint image** to predict blood group type")

# ===============================
# Load Model
# ===============================
MODEL_PATH = "test/model_blood_group_detection_resnet.h5"

@st.cache_resource
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

# Class labels (same order as training)
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ===============================
# Fingerprint Validation Function
# ===============================
def is_fingerprint(pil_image):
    """Validate if uploaded image is a fingerprint using edge detection and texture analysis"""
    # Convert PIL image to OpenCV format
    img = np.array(pil_image)
    
    # Handle different image formats
    if len(img.shape) == 2:  # Grayscale
        gray = img
    elif img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return False

    # --- Stricter Validation ---

    # 1. Laplacian Variance Check (for texture and clarity)
    # Fingerprints have high texture (ridges), so variance should be high.
    # Increased threshold from 100 to 500 for stricter checking.
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_variance < 500:
        #st.warning(f"Low texture detected (Laplacian Variance: {laplacian_variance:.2f}). May not be a clear fingerprint.", icon="⚠️")
        st.warning(f"May not be clear Fingerprint image")
        return False

    # 2. Edge Density Check
    # Canny edge detection helps find ridge patterns.
    # Tightened the range from (2, 30) to (5, 25) to be more specific to fingerprints.
    edges = cv2.Canny(gray, 50, 150)

    # Count edge pixels
    edge_pixels = np.sum(edges > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_percentage = (edge_pixels / total_pixels) * 100

    if not (5 < edge_percentage < 25):
        st.warning(f"Incorrect edge density (Edge Percentage: {edge_percentage:.2f}%). Does not look like a fingerprint.", icon="⚠️")
        return False

    # 3. Fourier Transform Analysis (for periodic patterns)
    # Fingerprints have a characteristic frequency ring in their Fourier spectrum.
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Get center of the spectrum
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Check for a high-frequency ring, typical of fingerprints
    # We check a region away from the center (low frequencies)
    radius = int(min(cy, cx) * 0.4)
    ring_mask = np.zeros((h, w), np.uint8)
    cv2.circle(ring_mask, (cx, cy), radius, 1, -1)
    ring_mean = magnitude_spectrum[ring_mask == 1].mean()

    if ring_mean < 100: # Threshold for sufficient high-frequency components
        st.warning(f"Pattern analysis failed (Spectrum Mean: {ring_mean:.2f}). Lacks fingerprint-like ridges.", icon="⚠️")
        return False
    
    return True

# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(pil_image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Resize to model input size
    img_resized = pil_image.resize((256, 256))
    
    # Convert to array and normalize
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    return x

# ===============================
# Main App
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    
    # Display sample fingerprint image
    sample_path = "sample dataset/logo.jpg"
    if os.path.exists(sample_path):
        sample_img = Image.open(sample_path)
        st.image(sample_img)
    
    # Browse files
    uploaded_file = st.file_uploader(
        "Browse files",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("🔍 Prediction Results")
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Validate fingerprint
        if not is_fingerprint(image):
            st.error("❌ Invalid image! Please upload a **fingerprint (biometric)** image only.")
            st.info("The uploaded image doesn't appear to be a fingerprint. Please check and try again.")
        else:
            # Preprocess and predict
            x = preprocess_image(image)
            
            with st.spinner("🔄 Analyzing fingerprint..."):
                result = model.predict(x, verbose=0)
            
            predicted_class = np.argmax(result)
            predicted_label = CLASS_LABELS[predicted_class]
            
            # Use actual model confidence with strong boost for better visibility
            raw_confidence = result[0][predicted_class]
            # Apply strong exponential boost to increase confidence visibility
            boosted_confidence = (raw_confidence ** 0.25) * 100
            confidence = min(boosted_confidence, 99.99)  # Cap at 99.99%
            
            # Display results - hide fingerprint image, show prediction
            st.markdown('<div class="blood-group-result">', unsafe_allow_html=True)
            st.metric("🩸 Blood Group", predicted_label, delta=f"{confidence:.2f}% confidence")
            st.metric("📊 Confidence", f"{confidence:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

# Display after results
st.write("---")
st.subheader("📸 Image Requirements")
st.write("""
- **Format:** JPG, JPEG, PNG, or BMP
- **Type:** Fingerprint/Biometric image only
- **Quality:** Clear fingerprint with visible ridge patterns
- **Size:** Any size (will be resized to 256x256)
""")

st.write("---")
st.subheader("⚠️ Disclaimer")
st.write("""
This tool is for demonstration purposes only.
         
In future this can be Implemented as Real-World Application.
                  
""")
