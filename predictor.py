# predictor.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def load_and_preprocess_image(image_path, img_size=(128, 128)):
    """
    Load and preprocess a single image for prediction
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_pokemon_top3(model, image_array, class_names):
    """
    Predict top 3 most likely Pokemon classes for a single image
    """
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array)[0]
    top3_indices = predictions.argsort()[-3:][::-1]  # Get indices of top 3 predictions
    
    results = []
    for idx in top3_indices:
        pokemon = class_names[idx]
        confidence = predictions[idx]
        results.append((pokemon, confidence))
    
    return results

def print_predictions(predictions):
    """
    Print predictions in a formatted way
    """
    print("\nTop 3 predictions:")
    print("=" * 40)
    for i, (pokemon, confidence) in enumerate(predictions, 1):
        print(f"{i}. {pokemon:<20} {confidence:.2%}")
    print("=" * 40)

if __name__ == "__main__":
    # Load the model and class names
    try:
        model = load_model(os.path.join('models', 'best_pokemon_model.keras'))
        class_names = np.load(os.path.join('data', 'class_names.npy'))
    except FileNotFoundError:
        print("Error: Model or class names not found. Please run data_loader.py and model_trainer.py first.")
        exit(1)
    
    # Make predictions on test images
    while True:
        image_name = input("\nEnter the name of your Pokemon (or 'q' to quit): ")
        
        if image_name.lower() == 'q':
            break
            
        image_path = f"test/{image_name}.png"
        
        try:
            image = load_and_preprocess_image(image_path)
            predictions = predict_pokemon_top3(model, image, class_names)
            print_predictions(predictions)
            
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
        except Exception as e:
            print(f"Error processing image: {str(e)}")