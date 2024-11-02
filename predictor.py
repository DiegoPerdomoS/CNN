# predictor.py
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

def load_and_preprocess_image(image_path, img_size=(128, 128)):
    """
    Load and preprocess a single image for prediction
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_pokemon(model, image_array, class_names):
    """
    Predict Pokemon class for a single image
    """
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Load the model and class names
    try:
        model = load_model(os.path.join('models', 'pokemon_model.keras'))
        class_names = np.load(os.path.join('data', 'class_names.npy'))
    except FileNotFoundError:
        print("Error: Model or class names not found. Please run data_loader.py and model_trainer.py first.")
        exit(1)
    
    # Example: Make predictions on some test images
    image_path = input("Enter the path to your Pokemon image: ")
    
    try:
        image = load_and_preprocess_image(image_path)
        predicted_class, confidence = predict_pokemon(model, image, class_names)
        print(f"\nPredicted Pokemon: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
    except Exception as e:
        print(f"Error processing image: {str(e)}")