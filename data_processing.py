import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "dataset" #"fake commercial aircraft (AI)"

print("Folder Exists:", os.path.exists(dataset_path))
print("Files:", os.listdir(dataset_path) if os.path.exists(dataset_path) else "Path incorrect")


subfolder = "test/real commercial aircraft" 
subfolder_path = os.path.join(dataset_path, subfolder)

#initilize "images"
images = []

if os.path.exists(subfolder_path):
    images = [f for f in os.listdir(subfolder_path) if f.endswith((".jpg"))]
    print(f"Images in '{subfolder}':", images[:50])
else:
    print("Subfolder path incorrect")
#preview first image - converting colour format
if images:
    img_path = os.path.join(subfolder_path, images[0])  
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Sample Image: {images[0]}")
    plt.show()
else:
    print("No images found in the subfolder")

#define dataset paths
train_dir = dataset_path 
test_dir = dataset_path

#data preprocessing (rescaling, normalizing pixel values)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#load training data (resizing images, multi-class classification)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

#load testing data
test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)