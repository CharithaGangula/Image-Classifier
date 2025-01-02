import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gradio as gr
from PIL import Image

# Paths to dataset
train_dir = '/content/drive/MyDrive/dataset/train'  # Path to training data
val_dir = '/content/drive/MyDrive/dataset/val'      # Path to validation data

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Load MobileNet base model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for transfer learning (output for 5 classes)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Define class labels (assuming you have 5 classes)
class_labels = ['Bike', 'Cat', 'Dog', 'Horse', 'Ship']  # Update this with your actual class labels

# Prediction function
def predict_and_visualize(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0

    predictions = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(predictions[0])

        # Get the predicted class label
    predicted_class_label = class_labels[predicted_class]

    original_img = Image.fromarray((image_array * 255).astype('uint8'))

    return f"Predicted Class: {predicted_class_label}", original_img

# Gradio interface
interface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(type="pil", label="Original Image")
    ],
    title="Lightweight Image Classifier",
    description="Upload an image to classify."
)

# Launch Gradio app
interface.launch(share=True)