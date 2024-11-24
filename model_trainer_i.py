import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

def load_data(data_dir='data'):
    """
    Load the processed data from files
    """
    data = {
        'X_train': np.load(os.path.join(data_dir, 'X_train.npy')),
        'y_train': np.load(os.path.join(data_dir, 'y_train.npy')),
        'X_val': np.load(os.path.join(data_dir, 'X_val.npy')),
        'y_val': np.load(os.path.join(data_dir, 'y_val.npy')),
        'X_test': np.load(os.path.join(data_dir, 'X_test.npy')),
        'y_test': np.load(os.path.join(data_dir, 'y_test.npy'))
    }
    class_names = np.load(os.path.join(data_dir, 'class_names.npy'))
    return data, class_names

def create_cnn_model(input_shape, num_classes):
    """
    Create a more stable CNN model with better initial learning capacity
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block - starting smaller
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),  # Reduced dropout
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(train_accuracies, val_accuracies, test_accuracies, 
                         train_losses, val_losses, test_losses):
    """
    Plot training, validation, and test accuracy/loss history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_accuracies, label='Training Accuracy')
    ax1.plot(val_accuracies, label='Validation Accuracy')
    ax1.plot(test_accuracies, label='Test Accuracy', linestyle='--')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(train_losses, label='Training Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.plot(test_losses, label='Test Loss', linestyle='--')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(data, class_names):
    """
    Train and evaluate the CNN model with more stable parameters
    """
    input_shape = data['X_train'].shape[1:]
    num_classes = len(class_names)
    
    model = create_cnn_model(input_shape, num_classes)
    
    # Simpler learning rate schedule
    initial_learning_rate = 0.001  # Back to standard learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_pokemon_model.keras')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_losses = []
    val_losses = []
    test_losses = []
    
    # Simplified data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    print("\nStarting training with test evaluation...")
    for epoch in range(20):  # Back to 20 epochs
        print(f"Epoch {epoch + 1}/20")
        
        # Apply data augmentation to training data
        X_train_aug = data_augmentation(data['X_train'], training=True)
        
        history = model.fit(
            X_train_aug, data['y_train'],
            epochs=1,
            batch_size=32,
            validation_data=(data['X_val'], data['y_val']),
            callbacks=callbacks,
            verbose=1
        )
        
        train_accuracies.append(history.history['accuracy'][0])
        val_accuracies.append(history.history['val_accuracy'][0])
        train_losses.append(history.history['loss'][0])
        val_losses.append(history.history['val_loss'][0])
        
        test_loss, test_accuracy = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Fixed early stopping check
        if len(val_losses) > 5:  # Changed from 7 to 5
            if all(val_losses[-5] < val_losses[-i] for i in range(1, 5)):
                print("\nEarly stopping triggered!")
                break
    
    print(f"\nFinal Test accuracy: {test_accuracies[-1]:.4f}")
    plot_training_history(train_accuracies, val_accuracies, test_accuracies, 
                         train_losses, val_losses, test_losses)
    
    return best_model_path

if __name__ == "__main__":
    # Load the processed data
    print("Loading processed data...")
    try:
        data, class_names = load_data()
    except FileNotFoundError:
        print("Error: Processed data not found. Please run data_loader.py first.")
        exit(1)
    
    # Train the model
    print("\nTraining model...")
    best_model_path = train_and_evaluate_model(data, class_names)
    print(f"\nBest model saved to {best_model_path}")