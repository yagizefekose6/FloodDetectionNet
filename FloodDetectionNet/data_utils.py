import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, img_height=224, img_width=224, num_augmentations=5):
        """
        Initialize the data preprocessor.
        
        Args:
            img_height: Height of input images
            img_width: Width of input images
            num_augmentations: Number of augmented samples to generate per image
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_augmentations = num_augmentations
        
        # Initialize data augmentation generator
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

    def _sort_files_by_number(self, files):
        """Sort files based on their numeric identifiers."""
        return dict(sorted(
            {int(re.findall(r'\d+', f)[0]): f for f in files}.items()
        ))

    def load_and_preprocess_data(self, images_path, masks_path):
        """
        Load and preprocess images and their corresponding masks.
        
        Args:
            images_path: Path to directory containing input images
            masks_path: Path to directory containing mask images
            
        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        # Get and sort file lists
        all_images = os.listdir(images_path)
        all_masks = os.listdir(masks_path)
        
        images_dict = self._sort_files_by_number(all_images)
        masks_dict = self._sort_files_by_number(all_masks)
        
        images_paths = [os.path.join(images_path, f) for f in images_dict.values()]
        masks_paths = [os.path.join(masks_path, f) for f in masks_dict.values()]
        
        images = []
        masks = []
        
        # Process each image-mask pair
        for img_path, mask_path in zip(images_paths, masks_paths):
            # Load and preprocess image
            image = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=(self.img_height, self.img_width)
            )
            input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            images.append(input_arr)
            
            # Load and preprocess mask
            mask = tf.keras.preprocessing.image.load_img(
                mask_path,
                target_size=(self.img_height, self.img_width),
                color_mode="grayscale"
            )
            input_mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0
            masks.append(input_mask)
            
            # Generate augmented samples
            for _ in range(self.num_augmentations):
                augmented_image = self.datagen.random_transform(input_arr)
                augmented_mask = self.datagen.random_transform(input_mask)
                images.append(augmented_image)
                masks.append(augmented_mask)
        
        return np.array(images), np.array(masks)

    def create_data_generator(self, images, masks, batch_size=32):
        """
        Create a data generator for training.
        
        Args:
            images: Input images array
            masks: Target masks array
            batch_size: Batch size for training
            
        Returns:
            Data generator yielding (image, mask) pairs
        """
        def generator():
            while True:
                indices = np.random.permutation(len(images))
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    yield images[batch_indices], masks[batch_indices]
        
        return generator() 