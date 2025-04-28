import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#extensive troubleshooting and trial and error taken place - code works but could be causing unstable test accuarcy 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

#define dataset paths
dataset_path = "dataset"  #main dataset directory
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

#set image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

#data preprocessing (resizing, rescaling, augmentation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#loading training data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

#loading validation data
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#loading test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

#building CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

#compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train the model
epochs = 10
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

#evaluate model on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")

#generate predictions
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

#classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

#confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#display sample predictions
def display_sample_predictions():
    sample_images, sample_labels = next(test_data)
    predictions = model.predict(sample_images)
    pred_classes = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(12, 6))
    for i in range(5): 
        plt.subplot(1, 5, i+1)
        plt.imshow(sample_images[i])
        plt.axis('off')
        plt.title(f"Pred:\n{list(test_data.class_indices.keys())[pred_classes[i]]}\n\nActual:\n{list(test_data.class_indices.keys())[np.argmax(sample_labels[i])]}",
                  fontsize=10) 
    plt.subplots_adjust(wspace=0.5)
    plt.show()

display_sample_predictions()

#Save class label order to a text file
label_map = train_data.class_indices
sorted_labels = sorted(label_map.items(), key=lambda x: x[1])
class_labels = [label for label, index in sorted_labels]

#Write into file
with open("class_labels.txt", "w") as f:
    for label in class_labels:
        f.write(label + "\n")

model.save("aircraft_classifier.keras")