# data_preprocessing.py

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
import argparse

from torchvision import transforms


# -------------------------------
# 4. Data Preprocessing
# -------------------------------

# -------------------------------
# 4.1. Parse Transcriptions
# -------------------------------

def parse_lines_transcription(lines_txt_path):
    """
    Parse the lines.txt file to extract line-level transcriptions.

    Args:
        lines_txt_path (str): Path to the lines.txt file.

    Returns:
        List of dictionaries with keys: 'form_id', 'line_id', 'transcription', 'image_filename'
    """
    transcription_data = []

    # Debugging: Print the path being accessed
    print(f"Attempting to open lines.txt at: {os.path.abspath(lines_txt_path)}")

    if not os.path.exists(lines_txt_path):
        raise FileNotFoundError(f"lines.txt not found at path: {os.path.abspath(lines_txt_path)}")

    with open(lines_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Parsing lines.txt"):
            # Skip header or comment lines
            if line.startswith('#'):
                continue  # Skip lines starting with '#'

            # Each valid line in lines.txt has the format:
            # a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
            parts = line.strip().split(' ')
            if len(parts) < 9:
                continue  # Skip malformed lines

            line_id = parts[0]  # e.g., a01-000u-00
            # parts[1] = 'ok' or 'err' (segmentation result), ignore for image mapping
            transcription = ' '.join(parts[8:]).strip()

            # Derive image filename from line_id
            # Assuming image filenames are like a01-000u-00.png
            image_filename = f"{line_id}.png"

            transcription_data.append({
                'form_id': line_id.split('-')[0],  # e.g., 'a01'
                'line_id': line_id,  # e.g., 'a01-000u-00'
                'transcription': transcription,
                'image_filename': image_filename
            })
    return transcription_data


def build_image_lookup(lines_dir):
    """
    Build a dictionary mapping image filenames to their full paths by searching recursively.
    This was done because the IAM dataset is nested with images and forms to the innermost OS directories.

    Args:
        lines_dir (str): Path to the lines/ directory containing line image files.

    Returns:
        Dict[str, str]: Mapping from image filename to its full path.
    """
    image_lookup = defaultdict(list)  # To handle potential duplicate filenames
    print(f"Building image lookup by searching recursively in: {os.path.abspath(lines_dir)}")
    for root, dirs, files in os.walk(lines_dir):
        for file in files:
            if file.lower().endswith('.png'):
                image_lookup[file].append(os.path.join(root, file))
    return image_lookup


def create_image_transcription_mapping(transcription_data, lines_dir):
    """
    Create a mapping between image filenames and their transcriptions by using a recursive search.

    Args:
        transcription_data (list): List of dictionaries containing transcription info.
        lines_dir (str): Path to the lines/ directory containing line image files.

    Returns:
        pandas.DataFrame: DataFrame with columns ['image_path', 'transcription']
    """
    image_lookup = build_image_lookup(lines_dir)
    data = []
    for entry in transcription_data:
        image_filename = entry['image_filename']
        if image_filename in image_lookup:
            if len(image_lookup[image_filename]) > 1:
                print(f"Warning: Multiple images found for {image_filename}. Using the first occurrence.")
            image_path = image_lookup[image_filename][0]  # Use the first occurrence
            data.append({
                'image_path': image_path,
                'transcription': entry['transcription']
            })
        else:
            print(
                f"Warning: Image file {image_filename} does not exist in any subdirectory of {os.path.abspath(lines_dir)}.")
    df = pd.DataFrame(data)
    return df


# -------------------------------
# 4.2. Image Preprocessing
# -------------------------------

def load_image(image_path):
    """
    Load an image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image or None: Loaded image or None if loading fails.
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def convert_to_grayscale(image):
    """
    Convert image to grayscale.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Grayscale image.
    """
    return image.convert('L')


def normalize_pixel_values(image):
    """
    Normalize pixel values to [0, 1].

    Args:
        image (PIL.Image): Grayscale image.

    Returns:
        numpy.ndarray: Normalized image array.
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    return image_np


def resize_image(image, fixed_height=128):
    """
    Resize image to a fixed height while maintaining aspect ratio.

    Args:
        image (PIL.Image): Grayscale image.
        fixed_height (int): Desired height.

    Returns:
        PIL.Image: Resized image.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_height = fixed_height
    new_width = int(aspect_ratio * new_height)

    # Handle Pillow versions
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS  # For older Pillow versions

    resized_image = image.resize((new_width, new_height), resample=resample_method)
    return resized_image


def pad_image(image, max_width, fixed_height=128):
    """
    Pad image to achieve consistent width.

    Args:
        image (PIL.Image): Resized grayscale image.
        max_width (int): Desired width after padding.
        fixed_height (int): Fixed height of the image.

    Returns:
        numpy.ndarray: Padded image array.
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    height, width = image_np.shape

    if width > max_width:
        # If image is wider than max_width, resize to max_width while maintaining height
        image_np = cv2.resize(image_np, (max_width, fixed_height))
    else:
        # Pad with zeros (black pixels) on the right
        pad_width = max_width - width
        image_np = np.pad(image_np, ((0, 0), (0, pad_width)), 'constant', constant_values=0.0)

    return image_np


def preprocess_images(df, fixed_height=128, resize_image_smaller=False, smaller_max_width=4096):
    """
    Preprocess all images: load, grayscale, normalize, resize, pad.

    Args:
        df (pandas.DataFrame): DataFrame with 'image_path' and 'transcription' columns.
        fixed_height (int): Desired image height.
        resize_image_smaller (bool): Whether to resize images to a smaller maximum width.
        smaller_max_width (int): The maximum width to resize images to if resize_image_smaller is True.

    Returns:
        numpy.ndarray: Array of preprocessed images.
        List[str]: List of corresponding transcriptions.
        int: Maximum width used for padding.
    """
    preprocessed_images = []
    transcriptions = []
    widths = []

    if resize_image_smaller:
        max_width = smaller_max_width
        print(f"Resizing images to fixed height {fixed_height}px and capping width at {max_width}px.")
    else:
        print("Resizing images to fixed height to determine maximum width...")

        # First pass: resize to fixed height and find maximum width
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="First pass - resizing images"):
            image = load_image(row['image_path'])
            if image is None:
                continue
            image = convert_to_grayscale(image)
            resized_image = resize_image(image, fixed_height)
            widths.append(resized_image.size[0])

        if not widths:
            raise ValueError("No images were successfully resized. Please check your image files.")

        max_width = max(widths)
        print(f"Maximum width after resizing: {max_width}px")

    print("Preprocessing images: resizing and padding...")

    # Second pass: resize to fixed height and pad to max_width
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Second pass - preprocessing images"):
        image = load_image(row['image_path'])
        if image is None:
            continue
        image = convert_to_grayscale(image)
        resized_image = resize_image(image, fixed_height)

        if resize_image_smaller:
            # Crop or pad the width to the smaller_max_width
            if resized_image.size[0] > smaller_max_width:
                # Resize to fixed height and smaller_max_width
                resized_image = resized_image.resize((smaller_max_width, fixed_height), Image.Resampling.LANCZOS)
            else:
                # Pad with zeros (black pixels) on the right
                pad_width = smaller_max_width - resized_image.size[0]
                image_np = np.array(resized_image).astype(np.float32) / 255.0
                image_np = np.pad(image_np, ((0, 0), (0, pad_width)), 'constant', constant_values=0.0)
                resized_image = Image.fromarray((image_np * 255).astype(np.uint8))
        else:
            # Pad to the maximum width determined earlier
            padded_image = pad_image(resized_image, max_width, fixed_height)
            resized_image = Image.fromarray((padded_image * 255).astype(np.uint8))

        # Convert to normalized numpy array
        image_np = normalize_pixel_values(resized_image)
        preprocessed_images.append(image_np)
        transcriptions.append(row['transcription'])

    if not preprocessed_images:
        raise ValueError("No images were successfully preprocessed. Please check your image files.")

    preprocessed_images = np.array(preprocessed_images)
    preprocessed_images = preprocessed_images[..., np.newaxis]  # Add channel dimension

    print(f"Preprocessed images shape: {preprocessed_images.shape}")

    if resize_image_smaller:
        final_max_width = smaller_max_width
    else:
        final_max_width = max_width

    return preprocessed_images, transcriptions, final_max_width


# -------------------------------
# 4.3. Label Encoding
# -------------------------------

def create_vocabulary(transcriptions, model_type='LSTM'):
    """
    Create a vocabulary of unique characters in the dataset.

    Args:
        transcriptions (List[str]): List of transcription strings.
        model_type (str): Target model type ('LSTM' or 'Transformer').

    Returns:
        Dict[str, int]: Mapping from character to unique integer index.
        Dict[int, str]: Mapping from index to character.
    """
    vocab = set()
    for text in transcriptions:
        vocab.update(list(text))
    vocab = sorted(list(vocab))

    # Add special tokens based on model type
    if model_type == 'Transformer':
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    else:  # LSTM
        special_tokens = ['<PAD>', '<UNK>']

    vocab = special_tokens + vocab  # Prepend special tokens

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Vocabulary size for {model_type}: {len(vocab)}")

    return char_to_idx, idx_to_char


def encode_transcriptions(transcriptions, char_to_idx, model_type='LSTM'):
    """
    Encode transcriptions into sequences of integer indices.

    Args:
        transcriptions (List[str]): List of transcription strings.
        char_to_idx (Dict[str, int]): Character to index mapping.
        model_type (str): Target model type ('LSTM' or 'Transformer').

    Returns:
        List[List[int]]: Encoded transcriptions.
    """
    encoded_transcriptions = []
    for text in transcriptions:
        if model_type == 'Transformer':
            # Add <SOS> at the beginning and <EOS> at the end
            encoded = [char_to_idx['<SOS>']] + [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text] + [
                char_to_idx['<EOS>']]
        else:  # LSTM
            encoded = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]
        encoded_transcriptions.append(encoded)
    return encoded_transcriptions


# -------------------------------
# Main Preprocessing Pipeline
# -------------------------------


def main_preprocessing_pipeline(ascii_dir, lines_dir, fixed_height=128, resize_image_smaller=False,
                                smaller_max_width=4096, model_type='LSTM'):
    """
    Execute the complete preprocessing pipeline.

    Args:
        ascii_dir (str): Path to the ascii/ directory.
        lines_dir (str): Path to the lines/ directory containing line image files.
        fixed_height (int): Desired image height.
        resize_image_smaller (bool): Whether to resize images to a smaller maximum width.
        smaller_max_width (int): The maximum width to resize images to if resize_image_smaller is True.
        model_type (str): Target model type ('LSTM' or 'Transformer').

    Returns:
        Dict: Dictionary containing preprocessed images, encoded labels, mappings, etc.
    """
    # 4.1. Parse Transcriptions
    lines_txt_path = os.path.join(ascii_dir, 'lines.txt')
    transcription_data = parse_lines_transcription(lines_txt_path)

    # 4.1. Step 2: Create Mapping between images and transcriptions
    df = create_image_transcription_mapping(transcription_data, lines_dir)
    print(f"Total valid image-transcription pairs: {df.shape[0]}")

    if df.empty:
        raise ValueError("DataFrame is empty after mapping. Please check your transcription and image files.")

    # 4.2. Image Preprocessing
    preprocessed_images, transcriptions, max_width = preprocess_images(
        df,
        fixed_height=fixed_height,
        resize_image_smaller=resize_image_smaller,
        smaller_max_width=smaller_max_width
    )

    # 4.3. Label Encoding
    char_to_idx, idx_to_char = create_vocabulary(transcriptions, model_type=model_type)
    encoded_transcriptions = encode_transcriptions(transcriptions, char_to_idx, model_type=model_type)

    # Optionally, save mappings for future use
    mappings = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'max_width': max_width,
        'fixed_height': fixed_height
    }

    # Save mappings with model type in filename
    mappings_filename = f'mappings_{model_type}.json'
    with open(mappings_filename, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    print(f"Mappings saved to {mappings_filename}")

    # Save encoded transcriptions with model type in filename
    encoded_transcriptions_filename = f'encoded_transcriptions_{model_type}.json'
    with open(encoded_transcriptions_filename, 'w', encoding='utf-8') as f:
        json.dump(encoded_transcriptions, f, ensure_ascii=False, indent=4)

    print(f"Encoded transcriptions saved to {encoded_transcriptions_filename}")

    # Save preprocessed images with model type in filename if needed
    # If preprocessed images are same for both models, you can omit this or keep it separate
    preprocessed_images_filename = f'preprocessed_images_{model_type}.npy'
    np.save(preprocessed_images_filename, preprocessed_images)
    print(f"Preprocessed images saved to {preprocessed_images_filename}")

    print("Preprocessing completed successfully.")

    return {
        'images': preprocessed_images,
        'transcriptions': transcriptions,
        'encoded_transcriptions': encoded_transcriptions,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'max_width': max_width,
        'fixed_height': fixed_height
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IAM Handwriting Dataset for OCR")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data/ directory')
    parser.add_argument('--fixed_height', type=int, default=128, help='Fixed height for image resizing')
    parser.add_argument('--resize_image_smaller', action='store_true', help='Resize images to a smaller maximum width')
    parser.add_argument('--smaller_max_width', type=int, default=4096, help='Maximum width for smaller resizing')

    # Define mutually exclusive group for model type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--LSTM', action='store_true', help='Preprocess data for LSTM model')
    group.add_argument('--transformer', action='store_true', help='Preprocess data for Transformer model')

    args = parser.parse_args()

    ascii_dir = os.path.join(args.data_dir, 'ascii')
    lines_dir = os.path.join(args.data_dir, 'lines')

    # Check if ascii_dir exists
    if not os.path.isdir(ascii_dir):
        raise NotADirectoryError(f"The ascii directory does not exist at path: {os.path.abspath(ascii_dir)}")

    # Check if lines_dir exists
    if not os.path.isdir(lines_dir):
        raise NotADirectoryError(f"The lines directory does not exist at path: {os.path.abspath(lines_dir)}")

    # Determine model type
    if args.LSTM:
        model_type = 'LSTM'
    elif args.transformer:
        model_type = 'Transformer'
    else:
        raise ValueError("Either --LSTM or --transformer must be specified.")

    # Execute preprocessing
    preprocessing_results = main_preprocessing_pipeline(
        ascii_dir=ascii_dir,
        lines_dir=lines_dir,
        fixed_height=args.fixed_height,
        resize_image_smaller=args.resize_image_smaller,
        smaller_max_width=args.smaller_max_width,
        model_type=model_type
    )

    # Note:
    # The preprocessed images are saved separately for each model type.
    # Depending on your training setup, you might want to handle this differently.
    # For example, if both models use the same images, you can save images without model type suffix.

    print("Preprocessed data saved to disk.")