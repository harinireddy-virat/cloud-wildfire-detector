import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

# --------------------
# Paths and Setup
# --------------------
MODEL_PATH = "wildfire_model.h5"
TRAIN_DIR = r"C:\Users\harin\ml_project\major2\forest_fire\Training and Validation"
TEST_DIR = r"C:\Users\harin\ml_project\major2\forest_fire\Testing"
IMAGE_PATH = r"C:\Users\harin\ml_project\major2\forest_fire\Testing\fire\abc176.jpg"

# --------------------
# Create Augmented Training & Validation Generators
# --------------------
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 🔥 split 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1/255)
test_dataset = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --------------------
# Define CNN Model (if retraining)
# --------------------
def build_model():
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),  # This is important to flatten the multi-dimensional output
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Adjust number of units based on your output
])

    model.summary()
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --------------------
# Load or Train Model
# --------------------
if os.path.exists(MODEL_PATH):
    print("✅ Loading existing model...")
    model = keras.models.load_model(MODEL_PATH)
else:
    print("🚀 Training new model...")
    model = build_model()

    # ✅ Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25,
        callbacks=[early_stop, checkpoint]
    )

# --------------------
# Evaluate Accuracy on Test Set
# --------------------
loss, accuracy = model.evaluate(test_dataset)
print(f"✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# --------------------
# Predict Single Image
# --------------------
def predictImage(filename):
    img1 = load_img(filename, target_size=(150, 150))
    img_array = img_to_array(img1)
    X = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(X)[0][0]
    print(f"Prediction value: {prediction:.4f}")

    label = "No Fire" if prediction >= 0.5 else "Fire"
    color = "green" if prediction >= 0.5 else "red"

    plt.figure(figsize=(6, 6), dpi=120)
    plt.imshow(img1)
    plt.title(f"{label} ({prediction:.2f})", fontsize=20, color=color)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------------------
# Run Prediction
# --------------------
if os.path.exists(IMAGE_PATH):
    predictImage(IMAGE_PATH)
else:
    print(f"❌ The file at {IMAGE_PATH} does not exist. Please check the path.") 