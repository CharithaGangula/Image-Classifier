
 Pre-Requisites:
- GPU: Requires an Nvidia GPU for efficient training (optional but recommended).
- Python 3.8+: Install the required Python version.
- TensorFlow: Install TensorFlow via `pip install tensorflow`.
- Gradio: Install Gradio via `pip install gradio`.
- Matplotlib: Install Matplotlib for plotting training results (`pip install matplotlib`).
- Pillow: Install Pillow for image processing (`pip install pillow`).

 Key Steps:
1. Import Necessary Libraries:
   - Import TensorFlow, MobileNet, ImageDataGenerator, and other required libraries.

2. Load Pre-Trained Model:
   - Load the MobileNet model pre-trained on ImageNet using `tensorflow.keras.applications.MobileNet`.

3. Define Data Generators:
   - Define `ImageDataGenerator` for preprocessing and augmenting the dataset.

4. Modify Model for Custom Dataset:
   - Add custom layers to the MobileNet model for transfer learning and output predictions for 5 classes.

5. Freeze Base Model Layers:
   - Freeze the layers of the pre-trained MobileNet base model to avoid retraining them.

6. Compile the Model:
   - Compile the model with the Adam optimizer and categorical cross-entropy loss function.

7. Train the Model:
   - Train the model on your dataset using the `fit()` method.

8. Plot Training Results:
   - Use Matplotlib to plot training and validation accuracy.

9. Define Prediction Function:
   - Create a function to preprocess input images and generate predictions using the trained model.

10. Integrate with UI (Gradio):
    - Integrate the classifier with a Gradio interface to allow users to upload images and see predictions.


Model Details:
- Model Architecture: MobileNet, pre-trained on ImageNet.
- Fine-Tuning: The model is fine-tuned to classify images into 5 custom classes: Bike, Cat, Dog, Horse, Ship.
- Training Data: The model is trained on a custom dataset split into training and validation directories.


Benefits of Using This Model:
- Lightweight: MobileNet is efficient and optimized for mobile devices, making it suitable for quick predictions.
- Transfer Learning: Leverages pre-trained weights to save time and resources.
- Easy to Use: The Gradio interface provides a simple way to interact with the model and make predictions.
