from tensorflow.keras.models import Sequential # type: ignore # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

# --------------------
# Paths
# --------------------
TRAIN_DIR = r"C:\Users\harin\ml_project\major2\forest_fire\Training and Validation"
TEST_DIR = r"C:\Users\harin\ml_project\major2\forest_fire\Testing"

# --------------------
# Data Preprocessing
# --------------------
img_size = (150, 150)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --------------------
# ANN Model
# --------------------
model_ann = Sequential([
    Flatten(input_shape=(150, 150, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------
# Training
# --------------------
model_ann.fit(train_data, epochs=10, validation_data=test_data)

# --------------------
# Evaluation
# --------------------
loss, accuracy = model_ann.evaluate(test_data)
print(f"✅ ANN Model Accuracy on Test Set: {accuracy * 100:.2f}%")
