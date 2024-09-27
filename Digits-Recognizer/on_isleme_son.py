
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image_color(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    # Additional contour merging or adjustment can be implemented here

    return bounding_box_image, bounding_boxes


def predict_digits(image_path, model):
    preprocessed_image = preprocess_image_color(image_path)
    bounding_box_image, bounding_boxes = detect_and_draw_bounding_boxes(preprocessed_image)
    predicted_digits = []

    for (x, y, w, h) in bounding_boxes:
        roi = preprocessed_image[y:y + h, x:x + w]
        roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (28, 28))
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        prediction = model.predict(roi)
        digit = np.argmax(prediction)
        predicted_digits.append(str(digit))
        cv2.putText(bounding_box_image, str(digit), (x + w // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    predicted_number = ''.join(predicted_digits)
    print("Tahmin Edilen Sayı: ", predicted_number)
    display_images(preprocessed_image, bounding_box_image)
    return bounding_box_image


def display_images(preprocessed_image, bounding_box_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Ön İşlenmiş Görüntü')
    plt.imshow(preprocessed_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Sınır Kutularıyla Görüntü')
    plt.imshow(cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=True)


def main(image_path, model_path):
    model = load_model(model_path)
    predicted_image = predict_digits(image_path, model)
    print("Tahminler tamamlandı, sonuçlar görselde gösterilmektedir.")




import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import urllib.request
import os

# Firebase Admin SDK'yı başlatın
cred = credentials.Certificate("C:/Users/ahmtk/OneDrive/Masaüstü/project-adf88-firebase-adminsdk-yyjwm-a947647c18.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'project-adf88.appspot.com'
})

def download_image_from_firebase(storage_path, local_path):
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    try:
        blob.download_to_filename(local_path)
        print(f"Dosya başarıyla indirildi: {local_path}")
    except Exception as e:
        print(f"Dosya indirme hatası: {e}")


def list_files_in_firebase_folder(folder_path):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=folder_path)
    return [blob.name for blob in blobs]


import time

def monitor_firebase_storage(folder_path, interval=60):
    processed_files = set()

    while True:
        try:
            current_files = set(list_files_in_firebase_folder(folder_path))
            new_files = current_files - processed_files

            for file_path in new_files:
                local_path = f"temp_{os.path.basename(file_path)}"
                download_image_from_firebase(file_path, local_path)
                # Burada yeni dosya ile işlem yapabilirsiniz (örneğin, model tahmini)
                main(local_path, model_path) # Örnek olarak ana fonksiyonu çağırabilirsiniz

            processed_files.update(new_files)
        except Exception as e:
            print(f"Error occurred: {e}")

        time.sleep(interval)

model_path = 'kendi_verilerimizz_model.h5'
# Dosyaları izlemeye başla
monitor_firebase_storage('images/', interval=5)


# Dosyayı indirin
firebase_image_path = 'images/cropped_image-6.png'
local_image_path = 'temp_image.png'
download_image_from_firebase(firebase_image_path, local_image_path)


# Test with your image
image_path = local_image_path

main(image_path, model_path)



