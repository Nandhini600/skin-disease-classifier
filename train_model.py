import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
BASE_DIR = 'dataset'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

df = pd.read_csv(os.path.join(BASE_DIR, 'HAM10000_metadata.csv'))
print(f"Total images: {len(df)}")

lesion_type_dict = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis',
    'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions', 'df': 'Dermatofibroma'
}
df['cell_type'] = df['dx'].map(lesion_type_dict)

def load_image(img_id):
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(BASE_DIR, part, f"{img_id}.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
    return None

print("\nLoading images...")
images, labels = [], []
for idx, row in df.iterrows():
    img = load_image(row['image_id'])
    if img is not None:
        images.append(img)
        labels.append(row['dx'])
    if (idx + 1) % 2000 == 0:
        print(f"Processed {idx + 1}/{len(df)}")
print(f"Loaded {len(images)} images")

images = np.array(images) / 255.0
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels = labels_encoded

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

y_train = to_categorical(y_train, 7)
y_val = to_categorical(y_val, 7)
y_test = to_categorical(y_test, 7)
print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("\nLoading MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('models/skin_disease_model.keras', monitor='val_accuracy', save_best_only=True)
]

print("\nTraining Phase 1 - Feature Extraction...")
history = model.fit(
    X_train, y_train,
    epochs=5, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks, class_weight=class_weights_dict, verbose=1
)

print("\nFine-tuning Phase 2...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    X_train, y_train,
    epochs=5, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks, class_weight=class_weights_dict, verbose=1
)

print("\nEvaluating...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
test_acc = accuracy_score(y_true, y_pred_classes)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_.tolist(), yticklabels=label_encoder.classes_.tolist())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('static/training_history.png', dpi=150)
plt.close()

print(f"\nDone! Model saved. Test Accuracy: {test_acc:.4f}")
