# Pokemon Image Classifier using CNN

## How to Run

1. **Data Preparation**:
   ```bash
   python data_loader.py
   ```
   This will download the Pokemon dataset and prepare it for training.

2. **Train the Model**:
   ```bash
   python model_trainer.py
   ```
   This trains the CNN model and saves the best model based on test accuracy.

2.1 **Train the Improved Model**:
   ```bash
   python model_trainer_i.py
   ```
   This trains the CNN model with improvements.

3. **Make Predictions**:
   ```bash
   python predictor.py
   ```
   When prompted, enter the path to your Pokemon image (e.g., "test/alakazam.png").
