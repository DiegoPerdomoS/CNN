# model_trainer.py
import os
import numpy as np
import matplotlib.pyplot as plt
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
    Create a CNN model for Pokemon classification
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
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
    Train and evaluate the CNN model with test set tracking
    """
    input_shape = data['X_train'].shape[1:]
    num_classes = len(class_names)
    
    model = create_cnn_model(input_shape, num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_pokemon_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_losses = []
    val_losses = []
    test_losses = []
    
    print("\nStarting training with test evaluation...")
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        
        history = model.fit(
            data['X_train'], data['y_train'],
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
    
    print(f"\nFinal Test accuracy: {test_accuracies[-1]:.4f}")
    plot_training_history(train_accuracies, val_accuracies, test_accuracies, 
                         train_losses, val_losses, test_losses)
    
    return model

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the processed data
    print("Loading processed data...")
    try:
        data, class_names = load_data()
    except FileNotFoundError:
        print("Error: Processed data not found. Please run data_loader.py first.")
        exit(1)
    
    # Train the model
    print("\nTraining model...")
    model = train_and_evaluate_model(data, class_names)
    
    # Save the trained model
    model_path = os.path.join('models', 'pokemon_model.keras')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")