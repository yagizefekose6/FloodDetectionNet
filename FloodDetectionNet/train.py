import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .model import build_unet_with_attention
from .data_utils import DataPreprocessor

def train_model(
    images_path,
    masks_path,
    output_dir='models',
    img_height=224,
    img_width=224,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    num_augmentations=5
):
    """
    Train the flood detection model.
    
    Args:
        images_path: Path to directory containing input images
        masks_path: Path to directory containing mask images
        output_dir: Directory to save model checkpoints
        img_height: Height of input images
        img_width: Width of input images
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data to use for validation
        num_augmentations: Number of augmented samples to generate per image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        img_height=img_height,
        img_width=img_width,
        num_augmentations=num_augmentations
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    images, masks = preprocessor.load_and_preprocess_data(images_path, masks_path)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=validation_split, random_state=42
    )
    
    # Create data generators
    train_generator = preprocessor.create_data_generator(X_train, y_train, batch_size)
    val_generator = preprocessor.create_data_generator(X_val, y_val, batch_size)
    
    # Build model
    print("Building model...")
    model = build_unet_with_attention(
        input_shape=(img_height, img_width, 3),
        output_channels=1
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model, history

def plot_training_history(history, output_dir):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        output_dir: Directory to save plots
    """
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == '__main__':
    # Example usage
    train_model(
        images_path='path/to/images',
        masks_path='path/to/masks',
        output_dir='models',
        epochs=100
    ) 