from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, storage
import os
import tempfile
import time
import asyncio
from typing import List, Dict
import base64
from io import BytesIO
from PIL import Image
import threading
import uvicorn
from pydantic import BaseModel

app = FastAPI(
    title="Digit Recognition API",
    description="Handwritten digit recognition service with Firebase integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
firebase_initialized = False

# Pydantic models
class PredictionResponse(BaseModel):
    predicted_number: str
    digits: List[str]
    bounding_boxes: List[Dict]
    confidence_scores: List[float]

class FirebaseConfig(BaseModel):
    credentials_path: str
    storage_bucket: str

# Firebase initialization
def initialize_firebase(credentials_path: str, bucket_name: str):
    global firebase_initialized
    try:
        if not firebase_initialized:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            firebase_initialized = True
        return True
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        return False

# Load model
def load_digit_model(model_path: str):
    global model
    try:
        model = load_model(model_path)
        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False

def preprocess_image_color(image_array):
    """Preprocess image for digit recognition"""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_array

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Erosion and Dilation
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    return dilated_image

def detect_and_draw_bounding_boxes(image):
    """Detect digits and create bounding boxes"""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        contour_area = cv2.contourArea(contour)

        # Relax filtering criteria
        if contour_area > 50 and w > 5 and h > 5 and 0.1 < aspect_ratio < 10.0:
            bounding_box_image = cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_boxes.append((x, y, w, h))

    # Sort bounding boxes by x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    return bounding_box_image, bounding_boxes

def predict_digits_from_array(image_array):
    """Predict digits from image array"""
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    preprocessed_image = preprocess_image_color(image_array)
    bounding_box_image, bounding_boxes = detect_and_draw_bounding_boxes(preprocessed_image)
    predicted_digits = []
    confidence_scores = []
    bbox_data = []

    for (x, y, w, h) in bounding_boxes:
        roi = preprocessed_image[y:y + h, x:x + w]
        roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (28, 28))
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        
        prediction = model.predict(roi)
        digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        predicted_digits.append(str(digit))
        confidence_scores.append(confidence)
        bbox_data.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        
        cv2.putText(bounding_box_image, str(digit), (x + w // 2, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    predicted_number = ''.join(predicted_digits)
    
    return {
        "predicted_number": predicted_number,
        "digits": predicted_digits,
        "bounding_boxes": bbox_data,
        "confidence_scores": confidence_scores,
        "processed_image": bounding_box_image
    }

# Firebase helper functions
def download_image_from_firebase(storage_path: str, local_path: str):
    """Download image from Firebase Storage"""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"Firebase download error: {e}")
        return False

def list_files_in_firebase_folder(folder_path: str):
    """List files in Firebase Storage folder"""
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=folder_path)
        return [blob.name for blob in blobs]
    except Exception as e:
        print(f"Firebase list error: {e}")
        return []

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    model_path = os.getenv("MODEL_PATH", "kendi_verilerimizz_model.h5")
    if os.path.exists(model_path):
        load_digit_model(model_path)
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file not found at {model_path}")

@app.get("/")
async def root():
    return {"message": "Digit Recognition API", "status": "running"}

@app.post("/initialize-firebase")
async def initialize_firebase_endpoint(config: FirebaseConfig):
    """Initialize Firebase with credentials"""
    success = initialize_firebase(config.credentials_path, config.storage_bucket)
    if success:
        return {"message": "Firebase initialized successfully"}
    else:
        raise HTTPException(status_code=500, detail="Firebase initialization failed")

@app.post("/predict-upload", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """Predict digits from uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Predict
        result = predict_digits_from_array(image)
        
        return PredictionResponse(
            predicted_number=result["predicted_number"],
            digits=result["digits"],
            bounding_boxes=result["bounding_boxes"],
            confidence_scores=result["confidence_scores"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-firebase")
async def predict_from_firebase(firebase_path: str):
    """Predict digits from Firebase Storage image"""
    if not firebase_initialized:
        raise HTTPException(status_code=400, detail="Firebase not initialized")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download from Firebase
        success = download_image_from_firebase(firebase_path, temp_path)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found in Firebase")
        
        # Read and predict
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = predict_digits_from_array(image)
        
        # Cleanup
        os.unlink(temp_path)
        
        return PredictionResponse(
            predicted_number=result["predicted_number"],
            digits=result["digits"],
            bounding_boxes=result["bounding_boxes"],
            confidence_scores=result["confidence_scores"]
        )
    
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/firebase/list-files")
async def list_firebase_files(folder_path: str):
    """List files in Firebase Storage folder"""
    if not firebase_initialized:
        raise HTTPException(status_code=400, detail="Firebase not initialized")
    
    files = list_files_in_firebase_folder(folder_path)
    return {"files": files}

@app.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks, folder_path: str, interval: int = 60):
    """Start monitoring Firebase folder for new images"""
    if not firebase_initialized:
        raise HTTPException(status_code=400, detail="Firebase not initialized")
    
    background_tasks.add_task(monitor_firebase_storage, folder_path, interval)
    return {"message": f"Started monitoring {folder_path} with {interval}s interval"}

async def monitor_firebase_storage(folder_path: str, interval: int = 60):
    """Background task to monitor Firebase storage"""
    processed_files = set()
    
    while True:
        try:
            current_files = set(list_files_in_firebase_folder(folder_path))
            new_files = current_files - processed_files
            
            for file_path in new_files:
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Download and process
                    if download_image_from_firebase(file_path, temp_path):
                        image = cv2.imread(temp_path)
                        if image is not None:
                            result = predict_digits_from_array(image)
                            print(f"Processed {file_path}: {result['predicted_number']}")
                        
                        os.unlink(temp_path)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            processed_files.update(new_files)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        await asyncio.sleep(interval)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "firebase_initialized": firebase_initialized
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )