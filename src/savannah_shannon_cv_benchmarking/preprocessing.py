"""
Image preprocessing module.
Handles image loading, resizing, channel conversion, and normalization.
"""

import numpy as np
from PIL import Image  # type: ignore[reportMissingImports]
from typing import List, Tuple, Union
from pathlib import Path


class ImagePreprocessor:
    # preprocess images to consistent format and size

    def __init__(self, image_size: int = 64, color_mode: str = 'rgb'):
        self.image_size = image_size
        self.color_mode = color_mode.lower()
        self.target_channels = 1 if self.color_mode == 'grayscale' else 3

    def load_and_preprocess(self,
                        image_data: Union[str, np.ndarray],
                        ) -> np.ndarray:
        if isinstance(image_data, str):
            img = Image.open(image_data)
        else:
            # Convert to uint8
            if image_data.max() <= 1:
                img_data = (image_data * 255).astype(np.uint8)
            else:
                img_data = image_data.astype(np.uint8)

            # squeeze single-channel images for PIL (PIL doesn't handle (H, W, 1))
            if img_data.ndim == 3 and img_data.shape[2] == 1:
                img_data = img_data.squeeze(axis=2)

            img = Image.fromarray(img_data)

        # convert to target color mode
        if self.color_mode == 'grayscale':
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')

        # resize with aspect ratio preservation
        img = self._resize_with_padding(img, self.image_size)

        # convert to numpy and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        return img_array

    def preprocess_batch(self,
                        image_data: Union[List[str], np.ndarray],
                        ) -> np.ndarray:
        if isinstance(image_data, list):
            # Load from paths
            images = []
            for img_path in image_data:
                try:
                    img = self.load_and_preprocess(img_path)
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to load {img_path}: {str(e)}")
        else:
            # handle numpy array input
            if len(image_data.shape) == 3:
                # (N, H, W) grayscale
                images = []
                for i in range(len(image_data)):
                    img = self.load_and_preprocess(image_data[i])
                    images.append(img)
            elif len(image_data.shape) == 4:
                # (N, H, W, C)
                images = []
                for i in range(len(image_data)):
                    img = self.load_and_preprocess(image_data[i])
                    images.append(img)
            else:
                raise ValueError(f"Invalid array shape: {image_data.shape}")

        # stack into batch
        batch = np.stack(images, axis=0)

        # ensure correct shape
        if self.color_mode == 'grayscale':
            if len(batch.shape) == 3:
                batch = np.expand_dims(batch, axis=-1)
            batch_shape = (len(batch), self.image_size, self.image_size, 1)
        else:
            if len(batch.shape) == 3:
                batch = np.expand_dims(batch, axis=-1)
                batch = np.tile(batch, (1, 1, 1, 3))
            batch_shape = (len(batch), self.image_size, self.image_size, 3)

        assert batch.shape == batch_shape, f"Shape mismatch: {batch.shape} != {batch_shape}"

        return batch

    def _resize_with_padding(self, img: Image.Image, target_size: int) -> Image.Image:
        # resize image with aspect ratio preservation using padding
        # calculate scaling factor
        width, height = img.size
        scale = min(target_size / width, target_size / height)

        # resize
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # create canvas and paste
        if img.mode == 'L':
            canvas = Image.new('L', (target_size, target_size), color=0)
        else:
            canvas = Image.new('RGB', (target_size, target_size), color=(0, 0, 0))

        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(img, (offset_x, offset_y))

        return canvas

    def create_flattened_features(self, image_batch: np.ndarray) -> np.ndarray:
        # flatten image batch for classical ML models
        batch_size = image_batch.shape[0]
        flattened = image_batch.reshape(batch_size, -1)
        return flattened
