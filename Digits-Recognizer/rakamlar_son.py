import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Veri seti yolunu ayarlama
train_dir = 'myenv/digit_database/Train'
test_dir = 'myenv/digit_database/Test'

# Eğitim ve test verilerini yükleme
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        for img_name in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, img_name)
            img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
            img = image.img_to_array(img)
            images.append(img)
            labels.append(int(label))
    return np.array(images), np.array(labels)

train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# Normalizasyon
train_images = train_images / 255.0
test_images = test_images / 255.0

# Veriyi yeniden şekillendirme
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# One-hot encoding
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Veri setini eğitim ve doğrulama olarak ayırma
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Modelin tanımlanması
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_regularizer=l2(0.001), input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), padding='Same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), padding='Same', activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])



optimizer = Adam(lr=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),  # Batch size'ı küçültmek
                    epochs=50, validation_data=(X_val, Y_val),
                    steps_per_epoch=X_train.shape[0] // 32,
                    callbacks=[early_stopping, reduce_lr])

model.save('kendi_verilerimizz_model.h5')


#####################
#ön işleme

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


# Test with your image
image_path = 'myenv/toplama/kayit.png'
model_path = 'kendi_verilerimizz_model.h5'
main(image_path, model_path)
