# data_loader.py
import kagglehub
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from sklearn.model_selection import train_test_split

def load_pokemon_dataset(base_path, img_size=(128, 128)):
    """
    Load Pokemon images and prepare them for training.
    """
    images = []
    labels = []
    class_names = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            class_names.append(dir_name)
            class_path = os.path.join(root, dir_name)
            class_idx = len(class_names) - 1
            
            print(f"Processing class: {dir_name}")
            
            # Process each image in the class directory
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(img_size)
                        img_array = img_to_array(img)
                        
                        images.append(img_array)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
    
    X = np.array(images)
    y = np.array(labels)
    X = X.astype('float32') / 255.0
    
    return X, y, class_names

def prepare_data(X, y, test_size=0.15, validation_split=0.15):
    """
    Split data into train, validation, and test sets.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=validation_split, 
        random_state=42,
        stratify=y_train_val
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

def save_data(data, class_names, save_dir='data'):
    """
    Save the processed data and class names to files
    """
    # Create data directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each array separately
    np.save(os.path.join(save_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(save_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(save_dir, 'X_val.npy'), data['X_val'])
    np.save(os.path.join(save_dir, 'y_val.npy'), data['y_val'])
    np.save(os.path.join(save_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(save_dir, 'y_test.npy'), data['y_test'])
    np.save(os.path.join(save_dir, 'class_names.npy'), class_names)

if __name__ == "__main__":
    print("Downloading dataset...")
    path = kagglehub.dataset_download("lantian773030/pokemonclassification")
    print("Loading and preprocessing images...")
    X, y, class_names = load_pokemon_dataset(path)
    print("Splitting dataset...")
    data = prepare_data(X, y)
    
    print(f"\nDataset statistics:")
    print(f"Total number of images: {len(X)}")
    print(f"Number of classes: {len(class_names)}")
    print("Classes:", class_names)
    print(f"\nTraining set shape: {data['X_train'].shape}")
    print(f"Validation set shape: {data['X_val'].shape}")
    print(f"Test set shape: {data['X_test'].shape}")

    # Save the processed data
    print("\nSaving processed data...")
    save_data(data, class_names)
    print("Data saved successfully!")
